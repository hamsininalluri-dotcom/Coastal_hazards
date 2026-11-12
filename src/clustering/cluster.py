"""Incident clustering: baseline (geohash + time window) and DBSCAN/HDBSCAN on a combined
spatial + temporal + semantic distance. Also builds the cluster summaries."""
import math

import numpy as np
from sklearn.cluster import DBSCAN

from src.common import SEVERITY_SCORE, SOURCE_CREDIBILITY, haversine_km

PARAMS = dict(eps_km=15.0, eps_hours=12.0, eps_sem=0.6, w_emb=0.5)
_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash(lat, lon, precision=4):
    lat_r, lon_r, bits, ch, even, out = [-90.0, 90.0], [-180.0, 180.0], 0, 0, True, ""
    while len(out) < precision:
        rng, v = (lon_r, lon) if even else (lat_r, lat)
        mid = (rng[0] + rng[1]) / 2
        if v > mid:
            ch, rng[0] = (ch << 1) | 1, mid
        else:
            ch, rng[1] = ch << 1, mid
        even, bits = not even, bits + 1
        if bits == 5:
            out, bits, ch = out + _B32[ch], 0, 0
    return out


def baseline_cluster(lat, lon, t_hours, precision=4, window_h=6.0):
    keys = {}
    return np.array([keys.setdefault((geohash(a, b, precision), int(t // window_h)), len(keys)) for a, b, t in zip(lat, lon, t_hours)])


def combined_distance(emb, type_probs, lat, lon, t_hours, p=PARAMS):
    """Normalised distance: <=1 means 'could be the same incident' on every axis jointly."""
    d_geo = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) / p["eps_km"]
    d_t = np.abs(t_hours[:, None] - t_hours[None, :]) / p["eps_hours"]
    tv = np.zeros_like(d_geo)  # total-variation distance between type distributions (looped: O(n^2) memory, not O(n^2 k))
    for k in range(type_probs.shape[1]):
        tv += 0.5 * np.abs(type_probs[:, k][:, None] - type_probs[:, k][None, :])
    d_sem = (p["w_emb"] * (1 - np.clip(emb @ emb.T, 0, 1)) + (1 - p["w_emb"]) * tv) / p["eps_sem"]
    return np.sqrt(d_geo ** 2 + d_t ** 2 + d_sem ** 2)


def cluster_reports(emb, type_probs, lat, lon, t_hours, algo="dbscan", params=None, min_samples=2):
    """Labels per report; noise points become their own singleton cluster (never silently dropped)."""
    p = {**PARAMS, **(params or {})}
    n = len(lat)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0])
    D = combined_distance(np.asarray(emb), np.asarray(type_probs), np.asarray(lat), np.asarray(lon), np.asarray(t_hours), p)
    if algo == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN
            lab = HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=1, copy=True).fit_predict(D)
        except Exception as e:  # sklearn/HDBSCAN edge-case failure -> graceful fallback
            print(f"[clustering] HDBSCAN failed ({type(e).__name__}); using DBSCAN")
            lab = DBSCAN(eps=1.0, min_samples=min_samples, metric="precomputed").fit_predict(D)
    else:
        lab = DBSCAN(eps=1.0, min_samples=min_samples, metric="precomputed").fit_predict(D)
    nxt = lab.max() + 1 if len(lab) else 0
    lab = lab.copy()
    for i in np.where(lab == -1)[0]:
        lab[i], nxt = nxt, nxt + 1
    return lab


def _components(idx, pairs):
    parent = {i: i for i in idx}

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        if a in parent and b in parent:
            parent[f(a)] = f(b)
    return {i: f(i) for i in idx}


def build_clusters(df, labels, dup_pairs=()):
    """df: enriched reports (aligned with labels) with columns report_id, latitude, longitude, timestamp, pred_type,
    confidence, severity_pred, source, media_flag, flags, contradictions. dup_pairs: iterable of (i, j) row indices."""
    out = []
    for lab in sorted(set(labels), key=lambda l: df.timestamp[np.asarray(labels) == l].min()):
        idx = [i for i, l in enumerate(labels) if l == lab]
        g = df.iloc[idx]
        comp = _components(idx, dup_pairs)
        n_unique = len(set(comp.values()))
        lat0, lon0 = g.latitude.mean(), g.longitude.mean()
        spread = float(np.sqrt(np.mean(haversine_km(g.latitude, g.longitude, lat0, lon0) ** 2)))
        w = g.confidence.clip(lower=0.05)
        tdist = {k: float(v) for k, v in w.groupby(g.pred_type).sum().div(w.sum()).items()}
        dom = max(tdist, key=tdist.get)
        agree = float((g.pred_type == dom).mean())
        cred = float(np.mean([SOURCE_CREDIBILITY.get(s, 0.4) + 0.1 * m for s, m in zip(g.source, g.media_flag)]))
        contra = [c for cs in g["contradictions"] for c in cs]
        contra_frac = float(np.mean([len(cs) > 0 for cs in g["contradictions"]]))
        rum_frac = float(np.mean(["rumour_language" in f for f in g["flags"]]))
        volume = min(1.0, math.log2(1 + n_unique) / math.log2(6))
        base = float(g.confidence.mean())
        conf = float(np.clip(0.35 * base + 0.25 * agree + 0.25 * volume + 0.15 * min(cred, 1) - 0.15 * contra_frac - 0.20 * rum_frac, 0, 1))
        if n_unique < 2:
            conf = min(conf, 0.45)  # one source (or re-posts of it) is never corroboration
        reasons = [f"{n_unique} distinct report(s) ({len(g)} incl. {len(g) - n_unique} duplicate(s))",
                   f"{agree:.0%} of reports agree on '{dom}'", f"mean classifier confidence {base:.2f}",
                   f"mean source credibility {min(cred, 1):.2f}"]
        if contra_frac:
            reasons.append(f"{contra_frac:.0%} of reports contain contradictions (-)")
        if rum_frac:
            reasons.append(f"{rum_frac:.0%} of reports use rumour-like language (-)")
        if n_unique < 2:
            reasons.append("single report: not corroborated")
        out.append(dict(
            cluster_id=f"C{len(out) + 1:04d}", report_ids=g.report_id.tolist(), n_reports=len(g), n_unique_reports=n_unique,
            duplicate_count=len(g) - n_unique, centroid_lat=float(lat0), centroid_lon=float(lon0), spread_km=spread,
            first_report_time=g.timestamp.min().isoformat(), last_report_time=g.timestamp.max().isoformat(),
            dominant_type=dom, type_distribution=tdist, confidence=conf, confidence_reasons=reasons,
            confidence_label="unconfirmed" if conf < 0.5 else "corroborated (still unverified)",
            severity_score=float(np.mean([SEVERITY_SCORE.get(s, 0) for s in g.severity_pred])),
            source_credibility=min(cred, 1.0), contradictions=sorted(set(contra))[:6], contradiction_count=len(contra),
            rumour_fraction=rum_frac, media_count=int(g.media_flag.sum()), sources=g.source.value_counts().to_dict()))
    return out
