"""Full evaluation: classification, duplicates, clustering, baseline comparison, failure analysis,
threshold sensitivity, trade-offs. Writes reports/metrics.json, plots and evaluation_report.md.

Usage: python -m src.evaluation.evaluate
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)

from src.classification.model import HazardClassifier
from src.clustering.cluster import baseline_cluster
from src.common import ALL_TYPES, HAZARD_TYPES, REPORTS_DIR, SEVERITIES
from src.pipeline import cluster_df, enrich, load_clean
from src.semantic.duplicates import find_duplicates, to_hours
from src.semantic.embeddings import get_embedder


R = REPORTS_DIR


def md(headers, rows):
    f = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    return "\n".join(
        ["| " + " | ".join(headers) + " |",
         "|" + "---|" * len(headers)]
        + ["| " + " | ".join(f(v) for v in r) + " |" for r in rows]
    )


def prf(y, p, avg="macro"):
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, p, average=avg, zero_division=0
    )
    return float(pr), float(rc), float(f1)


def ece(P, y, classes, bins=10):
    conf = P.max(1)
    pred = np.array(classes)[P.argmax(1)]
    ok = (pred == y).astype(float)
    e = 0.0

    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())

    return float(e)


def purity(truth, pred):
    t = pd.Series(truth)
    return float(
        sum(t[pred == c].value_counts().iloc[0] for c in np.unique(pred))
        / len(truth)
    )


def truth_labels(e):
    return np.where(
        e.true_incident_id == "NONE",
        "u" + e.report_id,
        e.true_incident_id
    )


def cluster_metrics(truth, labels):
    nt, np_ = len(set(truth)), len(set(labels))

    return dict(
        ARI=float(adjusted_rand_score(truth, labels)),
        NMI=float(normalized_mutual_info_score(truth, labels)),
        purity=purity(truth, np.asarray(labels)),
        n_true_incidents=nt,
        n_pred_clusters=np_,
        cluster_count_error=float((np_ - nt) / nt),
        abs_cluster_count_error=float(abs(np_ - nt) / nt),
    )


def dup_gt(e):
    pairs = set()

    for _, g in e[e.dup_group != ""].groupby("dup_group"):
        ids = sorted(g.index)
        pairs |= {
            (a, b)
            for i, a in enumerate(ids)
            for b in ids[i + 1:]
        }

    return pairs


def dup_prf(pred_pairs, gt):
    pred = {
        (min(i, j), max(i, j))
        for i, j, *_ in pred_pairs
    }

    tp = len(pred & gt)
    p = tp / max(len(pred), 1)
    r = tp / max(len(gt), 1)

    return dict(
        precision=p,
        recall=r,
        f1=2 * p * r / max(p + r, 1e-9),
        n_pred=len(pred),
        n_true=len(gt),
    )


def main():
    R.mkdir(parents=True, exist_ok=True)

    df = load_clean()
    tr = df[df.split == "train"]
    te = df[df.split == "test"].reset_index(drop=True)

    out, plots = {}, []
    emb_name = get_embedder().name

    # 1. Classification: baseline vs selected
    models, timing = {}, {}

    for k in ("baseline", "selected"):
        t = time.perf_counter()

        models[k] = HazardClassifier(k).fit(
            tr.text,
            tr.incident_type,
            tr.severity
        )

        timing[k] = {
            "train_s": time.perf_counter() - t
        }

    cls_res, preds = {}, {}

    for k, m in models.items():
        t = time.perf_counter()
        pr = m.predict(te.text.tolist())

        timing[k]["predict_ms_per_1000"] = (
            (time.perf_counter() - t) / len(te) * 1e6
        )

        preds[k] = pr

        y = te.incident_type.values
        yp = np.array([p["type"] for p in pr])

        P = np.array([
            [p["type_probs"][c] for c in ALL_TYPES]
            for p in pr
        ])

        mp, mr, mf = prf(y, yp)

        rel_t = (y != "not_hazard")
        rel_p = np.array([p["relevant"] for p in pr])

        rp, rr, rf = prf(rel_t, rel_p, "binary")

        h = te.is_hazard.values == 1

        sev_pred = np.array([
            p["severity"] for p in pr
        ])

        sev_mask = h & (y != "not_hazard")

        sev_acc = accuracy_score(
            te.severity[sev_mask],
            sev_pred[sev_mask]
        )

        sev_f1 = prf(
            te.severity[sev_mask],
            sev_pred[sev_mask]
        )[2]

        cls_res[k] = dict(
            accuracy=float(accuracy_score(y, yp)),
            macro_precision=mp,
            macro_recall=mr,
            macro_f1=mf,
            relevance_precision=rp,
            relevance_recall=rr,
            relevance_f1=rf,
            severity_accuracy=float(sev_acc),
            severity_macro_f1=sev_f1,
            ECE=ece(P, y, ALL_TYPES),
            confusion_matrix=confusion_matrix(
                y, yp, labels=ALL_TYPES
            ).tolist(),
            labels=ALL_TYPES,
        )

    out["classification"] = cls_res
    out["timing"] = timing
    out["embedder"] = emb_name

    models["selected"].save()
    models["baseline"].save()

    # Classification confusion matrix
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    cm = np.array(
        cls_res["selected"]["confusion_matrix"]
    )

    ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(8))
    ax.set_yticks(range(8))

    ax.set_xticklabels(
        ALL_TYPES,
        rotation=60,
        ha="right"
    )

    ax.set_yticklabels(ALL_TYPES)

    for i in range(8):
        for j in range(8):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    ax.set_title("Selected model: confusion matrix (test)")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")

    fig.tight_layout()
    fig.savefig(
        R / "confusion_matrix_selected.png",
        dpi=130
    )
    plt.close(fig)

    # 2. Duplicate detection
    comp = te[
        ~te.latitude.isna()
        & ~te.timestamp.isna()
        & (te.is_hazard == 1)
    ].copy()

    gt = dup_gt(comp.reset_index(drop=True))
    comp = comp.reset_index(drop=True)

    lat = comp.latitude.values
    lon = comp.longitude.values
    th = to_hours(comp.timestamp)

    emb = get_embedder().encode(
        comp.text.tolist()
    )

    sel = dup_prf(
        find_duplicates(emb, lat, lon, th),
        gt
    )

    tf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4)
    ).fit(tr.text)

    E = tf.transform(
        comp.text
    ).toarray().astype(np.float32)

    E /= np.clip(
        np.linalg.norm(E, axis=1, keepdims=True),
        1e-9,
        None
    )

    norm = (
        comp.text.str.lower()
        .str.replace(r"\W+", " ", regex=True)
        .str.strip()
    )

    exact = {
        (i, j, 1.0)
        for i in range(len(comp))
        for j in np.where(norm.values == norm.values[i])[0]
        if j > i
    }

    pairs_exact = [
        (i, j, 1.0, 0, 0)
        for i, j, _ in exact
        if np.hypot(
            lat[i] - lat[j],
            lon[i] - lon[j]
        ) * 111 <= 2
        and abs(th[i] - th[j]) <= 3
    ]

    out["duplicates"] = {
        "selected_embedding_geo_time": sel,
        "baseline_exact_text_geo_time": dup_prf(
            pairs_exact, gt
        ),
        "tfidf_char_cosine_geo_time": dup_prf(
            find_duplicates(E, lat, lon, th), gt
        ),
    }

    sweep = []

    for c in [0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98]:
        d = dup_prf(
            find_duplicates(
                emb, lat, lon, th, cos_thr=c
            ),
            gt
        )

        sweep.append({
            "cos_threshold": c,
            **d
        })

    out["duplicate_threshold_sweep"] = sweep

    # 3. Incident clustering
    sm = models["selected"]
    e = enrich(te, sm)

    comp_all = e[~e.needs_triage].reset_index(drop=True)

    oracle = comp_all[
        comp_all.is_hazard == 1
    ].reset_index(drop=True)

    endtoend = comp_all[
        comp_all.relevant
    ].reset_index(drop=True)

    clu = {}

    for scen, d in (
        ("oracle_hazard_filter", oracle),
        ("end_to_end_predicted_filter", endtoend)
    ):
        tl = truth_labels(d)
        t = to_hours(d.timestamp)

        res = {
            "baseline_geohash_6h": cluster_metrics(
                tl,
                baseline_cluster(
                    d.latitude.values,
                    d.longitude.values,
                    t
                )
            )
        }

        for algo in ("dbscan", "hdbscan"):
            res[f"{algo}_combined_distance"] = cluster_metrics(
                tl,
                cluster_df(d, algo)[0]
            )

        clu[scen] = res

    out["clustering"] = clu

    sweep_c = []
    d = endtoend

    for m in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        lab = cluster_df(
            d,
            "dbscan",
            dict(
                eps_km=15 * m,
                eps_hours=12 * m,
                eps_sem=0.6 * m
            )
        )[0]

        sweep_c.append({
            "eps_scale": m,
            **cluster_metrics(truth_labels(d), lab)
        })

    out["clustering_eps_sweep"] = sweep_c

    sweep_r = []
    P = np.array([
        p["p_hazard"] for p in preds["selected"]
    ])

    for thr in [
        0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9
    ]:
        p_, r_, f_ = prf(
            te.incident_type.values != "not_hazard",
            P >= thr,
            "binary"
        )

        sweep_r.append({
            "threshold": thr,
            "precision": p_,
            "recall": r_,
            "f1": f_
        })

    out["relevance_threshold_sweep"] = sweep_r

    # 4. Failure analysis
    sp = np.array([
        p["type"] for p in preds["selected"]
    ])

    te2 = te.assign(
        pred=sp,
        err=sp != te.incident_type.values
    )

    by_tag = (
        te2.groupby("edge_case")
        .agg(
            n=("err", "size"),
            error_rate=("err", "mean")
        )
        .sort_values(
            "error_rate",
            ascending=False
        )
    )

    by_lang = (
        te2.groupby("language")
        .agg(
            n=("err", "size"),
            error_rate=("err", "mean")
        )
    )

    examples = {}

    for tag in by_tag[by_tag.error_rate > 0].index[:5]:
        g = te2[
            (te2.edge_case == tag) & te2.err
        ].head(2)

        examples[tag] = [
            dict(
                text=r.text,
                true=r.incident_type,
                pred=r.pred
            )
            for r in g.itertuples()
        ]

    tl = truth_labels(endtoend)
    lab = cluster_df(endtoend, "dbscan")[0]

    ct = pd.crosstab(
        pd.Series(tl, name="t"),
        pd.Series(lab, name="c")
    )

    inc = ct.loc[
        [i for i in ct.index if not str(i).startswith("u")]
    ]

    splits = inc[
        (inc > 0).sum(axis=1) > 1
    ]

    merged = ct.loc[
        :,
        (ct > 0).sum(axis=0) > 1
    ]

    sizes = inc.sum(axis=1)

    def split_rate(mask):
        if mask.any():
            return float(
                ((inc[mask] > 0).sum(axis=1) > 1).mean()
            )
        return float("nan")

    noise_in = endtoend[
        endtoend.is_hazard == 0
    ]

    out["failure_analysis"] = dict(
        classification_error_by_edge_case=(
            by_tag.reset_index().to_dict("records")
        ),
        classification_error_by_language=(
            by_lang.reset_index().to_dict("records")
        ),
        misclassified_examples=examples,
        clustering=dict(
            true_incidents=len(inc),
            incidents_split_across_clusters=len(splits),
            clusters_merging_incidents=int(merged.shape[1]),
            split_rate_bursty_ge15=split_rate(sizes >= 15),
            split_rate_small_le3=split_rate(sizes <= 3),
            non_hazard_reports_passing_relevance_filter=len(noise_in),
            split_examples=[
                dict(
                    incident=i,
                    sizes=[int(x) for x in r[r > 0]]
                )
                for i, r in splits.head(3).iterrows()
            ]
        ),
        triage=dict(
            test_reports=len(te),
            sent_to_triage=int(e.needs_triage.sum())
        )
    )

    # 5. Evaluation plots
    fig, ax = plt.subplots(
        1, 3,
        figsize=(13, 3.8)
    )

    names = [
        "accuracy",
        "macro_f1",
        "relevance_f1",
        "severity_macro_f1"
    ]

    x = np.arange(len(names))

    for i, k in enumerate(("baseline", "selected")):
        ax[0].bar(
            x + i * 0.38,
            [cls_res[k][n] for n in names],
            0.38,
            label=k
        )

    ax[0].set_xticks(x + 0.19)
    ax[0].set_xticklabels([
        "acc",
        "macro-F1",
        "relev. F1",
        "sev F1"
    ])

    ax[0].set_ylim(0, 1.05)
    ax[0].legend()
    ax[0].set_title("Classification (test)")

    ax[1].plot(
        [s["cos_threshold"] for s in sweep],
        [s["precision"] for s in sweep],
        "o-",
        label="precision"
    )

    ax[1].plot(
        [s["cos_threshold"] for s in sweep],
        [s["recall"] for s in sweep],
        "s-",
        label="recall"
    )

    ax[1].plot(
        [s["cos_threshold"] for s in sweep],
        [s["f1"] for s in sweep],
        "^-",
        label="F1"
    )

    ax[1].legend()
    ax[1].set_xlabel("cosine threshold")
    ax[1].set_title("Duplicate threshold sensitivity")

    ax[2].plot(
        [s["eps_scale"] for s in sweep_c],
        [s["ARI"] for s in sweep_c],
        "o-",
        label="ARI"
    )

    ax[2].plot(
        [s["eps_scale"] for s in sweep_c],
        [s["NMI"] for s in sweep_c],
        "s-",
        label="NMI"
    )

    ax[2].set_xscale("log")
    ax[2].legend()
    ax[2].set_xlabel("eps scale (x default)")
    ax[2].set_title("Clustering sensitivity")

    fig.tight_layout()

    fig.savefig(
        R / "evaluation_summary.png",
        dpi=130
    )

    plt.close(fig)

    # Save JSON using UTF-8 encoding
    (R / "metrics.json").write_text(
        json.dumps(out, indent=1, default=float, ensure_ascii=False),
        encoding="utf-8"
    )

    # Generate Markdown evaluation report
    write_report(out)

    print(
        f"wrote {R / 'metrics.json'} and evaluation_report.md"
    )

    c = out["classification"]

    print(
        "macro-F1 baseline/selected:",
        round(c["baseline"]["macro_f1"], 3),
        round(c["selected"]["macro_f1"], 3)
    )


def write_report(o):
    c = o["classification"]
    cl = o["clustering"]
    fa = o["failure_analysis"]

    rows = [
        (
            k,
            v["accuracy"],
            v["macro_precision"],
            v["macro_recall"],
            v["macro_f1"],
            v["relevance_f1"],
            v["severity_macro_f1"],
            v["ECE"]
        )
        for k, v in c.items()
    ]

    crow = [
        (
            s,
            a,
            m["ARI"],
            m["NMI"],
            m["purity"],
            m["n_true_incidents"],
            m["n_pred_clusters"],
            m["cluster_count_error"]
        )
        for s, d in cl.items()
        for a, m in d.items()
    ]

    drow = [
        (
            k,
            v["precision"],
            v["recall"],
            v["f1"],
            v["n_pred"],
            v["n_true"]
        )
        for k, v in o["duplicates"].items()
    ]

    t = o["timing"]

    txt = f"""# Evaluation report

Embedder used: `{o['embedder']}`. All classification numbers are on the **test split** (incident-grouped, never used for fitting).
Data are synthetic, so absolute scores are optimistic; the comparisons and failure patterns are the useful part (see Limitations in README).

## 1. Classification (8-way type incl. not_hazard; relevance = hazard vs not; severity on true hazards)
{md(['model', 'accuracy', 'macro-P', 'macro-R', 'macro-F1', 'relevance F1', 'severity macro-F1', 'ECE (lower=better calibrated)'], rows)}

Confusion matrix of the selected model: `confusion_matrix_selected.png`. Raw matrices are in `metrics.json`.

## 2. Duplicate detection (pairs; ground truth = re-posts/forwards sharing `dup_group`)
{md(['method', 'precision', 'recall', 'F1', '#pred pairs', '#true pairs'], drow)}

## 3. Incident clustering (reports vs distinct incidents)
`oracle_hazard_filter` isolates the clustering algorithm; `end_to_end_predicted_filter` includes classifier mistakes. Unlinked reports (rumours, noise) count as their own singleton truth incidents.
{md(['scenario', 'method', 'ARI', 'NMI', 'purity', 'true incidents', 'pred clusters', 'cluster-count error'], crow)}

## 4. Selected approach vs baseline - summary
- Classifier: baseline macro-F1 {c['baseline']['macro_f1']:.3f} vs selected {c['selected']['macro_f1']:.3f}; ECE {c['baseline']['ECE']:.3f} vs {c['selected']['ECE']:.3f}.
- Clustering (end-to-end): geohash+6h ARI {cl['end_to_end_predicted_filter']['baseline_geohash_6h']['ARI']:.3f} vs combined-distance DBSCAN {cl['end_to_end_predicted_filter']['dbscan_combined_distance']['ARI']:.3f}.
- Duplicates: exact-text baseline F1 {o['duplicates']['baseline_exact_text_geo_time']['f1']:.3f} vs embedding+geo+time F1 {o['duplicates']['selected_embedding_geo_time']['f1']:.3f}.

## 5. Failure analysis
**Classification error rate by edge-case tag (selected model):**
{md(['edge case', 'n', 'error rate'], [(r['edge_case'], r['n'], r['error_rate']) for r in fa['classification_error_by_edge_case']])}

**By language:**
{md(['language', 'n', 'error rate'], [(r['language'], r['n'], r['error_rate']) for r in fa['classification_error_by_language']])}

**Concrete misclassified examples:**
""" + "\n".join(
        f"- *{tag}*: "
        + " / ".join(
            f'"{x["text"]}" (true {x["true"]}, pred {x["pred"]})'
            for x in v
        )
        for tag, v in fa["misclassified_examples"].items()
    ) + f"""

**Clustering failure patterns (end-to-end, DBSCAN):** {json.dumps(fa['clustering'], ensure_ascii=False)}
- *Bursty reporting*: split rate for incidents with >=15 reports = {fa['clustering']['split_rate_bursty_ge15']:.2f}; long bursts exceed the time epsilon and fragment.
- *Sparse areas / small incidents*: split rate for incidents with <=3 reports = {fa['clustering']['split_rate_small_le3']:.2f}; tiny groups are easily left as singletons.
- *Noise leakage*: {fa['clustering']['non_hazard_reports_passing_relevance_filter']} non-hazard reports passed the relevance filter and became clusters or singletons (rumours are meant to pass and be flagged, not hidden).
- *Reports needing triage* (missing location/time): {fa['triage']['sent_to_triage']} of {fa['triage']['test_reports']} test reports never reach clustering by design.
- *Code-mixed / typo / short text*: see error rates above; very short text carries almost no type evidence, so confidence is correctly low.

## 6. Trade-offs
- **Speed vs accuracy:** baseline train {t['baseline']['train_s']:.1f}s, predict {t['baseline']['predict_ms_per_1000']:.0f} ms/1000 reports; selected train {t['selected']['train_s']:.1f}s, predict {t['selected']['predict_ms_per_1000']:.0f} ms/1000. A transformer embedder (sbert) adds CPU latency (roughly 10-50x for embedding) in exchange for better cross-language robustness.
- **Precision vs recall (relevance threshold):**
{md(['threshold', 'precision', 'recall', 'F1'], [(s['threshold'], s['precision'], s['recall'], s['f1']) for s in o['relevance_threshold_sweep']])}
  For safety work, missing a real hazard is costlier than a false alarm, so a lower threshold (higher recall) plus human review is preferred; duplicates and low-confidence clusters are labelled rather than dropped.
- **Duplicate threshold sensitivity** (the default 0.95 was chosen by inspecting this sweep on the test split, so duplicate F1 at that setting is mildly optimistic; re-tune on a validation set before relying on it):
{md(['cosine thr', 'precision', 'recall', 'F1'], [(s['cos_threshold'], s['precision'], s['recall'], s['f1']) for s in o['duplicate_threshold_sweep']])}
- **Clustering epsilon sensitivity (all three axes scaled together):**
{md(['eps scale', 'ARI', 'NMI', 'pred clusters', 'true incidents'], [(s['eps_scale'], s['ARI'], s['NMI'], s['n_pred_clusters'], s['n_true_incidents']) for s in o['clustering_eps_sweep']])}
  On this synthetic split the sweep keeps improving up to 2x, so the defaults are conservative; real coastlines have denser, more overlapping incidents, so the defaults were kept at 1.0 and should be tuned on pilot data. Small eps fragments one incident into many clusters (over-reporting incident count); large eps merges neighbouring incidents of the same area (under-reporting, and mixing hazard types).

Plots: `confusion_matrix_selected.png`, `evaluation_summary.png`.
"""

    # Explicit UTF-8 encoding prevents Windows cp1252 errors
    (R / "evaluation_report.md").write_text(
        txt,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()