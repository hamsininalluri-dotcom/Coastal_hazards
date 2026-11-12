"""End-to-end pipeline: enrich (classify + semantic checks) -> duplicates -> clustering -> priority.
Python API for the evaluator, the REST service and the demo.  CLI: python -m src.pipeline"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.classification.fallback import KeywordClassifier
from src.classification.model import HazardClassifier
from src.clustering.cluster import build_clusters, cluster_reports
from src.common import ALL_TYPES, DATA_DIR, HAZARD_TYPES, MODELS_DIR
from src.priority.agent import assess
from src.semantic.checks import check_report
from src.semantic.duplicates import find_duplicates, to_hours
from src.semantic.embeddings import get_embedder


def load_clean(path=None):
    df = pd.read_csv(path or DATA_DIR / "processed" / "reports_clean.csv", keep_default_na=False)
    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df.timestamp.replace("", np.nan), errors="coerce")
    return df


def load_classifier(kind="selected"):
    """Load trained model; on ANY failure fall back to keyword rules (flagged in output as model_used)."""
    try:
        return HazardClassifier.load(kind)
    except Exception as e:
        print(f"[pipeline] model '{kind}' unavailable ({type(e).__name__}: {e}); using keyword fallback")
        return KeywordClassifier()


def enrich(df, clf, embedder=None):
    """Adds prediction + semantic-check columns. Never raises on model failure (falls back)."""
    df = df.copy().reset_index(drop=True)
    embedder = embedder or get_embedder()
    try:
        preds = clf.predict(df.text.tolist())
    except Exception as e:
        print(f"[pipeline] classifier failed ({type(e).__name__}); keyword fallback")
        preds = KeywordClassifier().predict(df.text.tolist())
    df["pred_type"] = [p["type"] for p in preds]
    df["p_hazard"] = [p["p_hazard"] for p in preds]
    df["relevant"] = [p["relevant"] for p in preds]
    df["severity_pred"] = [p["severity"] for p in preds]
    df["confidence"] = [p["confidence"] for p in preds]
    df["model_used"] = [p["model_used"] for p in preds]
    df["type_probs"] = [p["type_probs"] for p in preds]
    chk = [check_report(r.text, r.latitude, r.longitude, r.location_text, r.timestamp) for r in df.itertuples()]
    df["flags"] = [c["flags"] for c in chk]
    df["contradictions"] = [c["contradictions"] for c in chk]
    df["rumour_score"] = [c["rumour_score"] for c in chk]
    df["emb"] = list(embedder.encode(df.text.tolist()))
    df["needs_triage"] = df.latitude.isna() | df.longitude.isna() | df.timestamp.isna()
    return df


def probs_matrix(df):
    """Hazard-type probability matrix, renormalised over hazard types (not_hazard removed)."""
    M = np.array([[p.get(t, 0.0) for t in HAZARD_TYPES] for p in df.type_probs])
    return M / np.clip(M.sum(1, keepdims=True), 1e-9, None)


def cluster_df(df, algo="dbscan", params=None, dup_kwargs=None):
    """df: enriched reports that are relevant and complete. Returns (labels, clusters, dup_pairs)."""
    df = df.reset_index(drop=True)
    if df.empty:
        return np.array([], dtype=int), [], []
    t = to_hours(df.timestamp)
    emb = np.vstack(df.emb.to_list())
    pairs = find_duplicates(emb, df.latitude.to_numpy(float), df.longitude.to_numpy(float), t, **(dup_kwargs or {}))
    labels = cluster_reports(emb, probs_matrix(df), df.latitude.to_numpy(float), df.longitude.to_numpy(float), t, algo, params)
    clusters = build_clusters(df, labels, [(i, j) for i, j, *_ in pairs])
    return labels, clusters, pairs


def run(df, clf=None, algo="dbscan", now=None):
    """Full pipeline on a raw/clean report frame."""
    clf = clf or load_classifier()
    e = enrich(df, clf)
    triage = e[e.needs_triage]
    ok = e[(~e.needs_triage) & e.relevant]
    labels, clusters, pairs = cluster_df(ok, algo)
    now = now or (ok.timestamp.max().to_pydatetime() if len(ok) else datetime.utcnow())
    for c in clusters:
        c.update(assess(c, now))
        c["review_status"] = "Unverified"
    return dict(reports=e, triage=triage, clusters=clusters, dup_pairs=pairs, labels=labels,
                summary=dict(n_reports=len(e), n_relevant=int(e.relevant.sum()), n_triage=len(triage),
                             n_clusters=len(clusters), n_multi_report_incidents=sum(c["n_unique_reports"] >= 2 for c in clusters)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "processed" / "reports_clean.csv"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--algo", default="dbscan", choices=["dbscan", "hdbscan"])
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    df = pd.read_csv(a.data, keep_default_na=False, parse_dates=["timestamp"])
    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df.timestamp, errors="coerce")
    if a.split != "all":
        df = df[df.split == a.split]
    r = run(df, algo=a.algo)
    print(json.dumps(r["summary"], indent=1))
    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    for c in sorted(r["clusters"], key=lambda c: (order[c["priority"]], -c["n_unique_reports"]))[:a.top]:
        print(f"{c['cluster_id']} [{c['priority']}] {c['dominant_type']} reports={c['n_reports']} distinct={c['n_unique_reports']} "
              f"conf={c['confidence']:.2f} ({c['confidence_label']}) status=REPORTED/{c['review_status']}")


if __name__ == "__main__":
    main()
