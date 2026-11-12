"""Duplicate detection: cosine similarity on embeddings + geo-distance + time-window rules."""
import argparse

import numpy as np
import pandas as pd

from src.common import DATA_DIR, haversine_km
from src.semantic.embeddings import get_embedder

DEFAULTS = dict(cos_thr=0.95, max_km=2.0, max_hours=3.0)


def find_duplicates(emb, lat, lon, t_hours, cos_thr=DEFAULTS["cos_thr"], max_km=DEFAULTS["max_km"], max_hours=DEFAULTS["max_hours"]):
    """Return list of (i, j, cosine, km, hours) with i<j. Pairs lacking location or time are never matched."""
    emb, lat, lon, t = map(np.asarray, (emb, lat, lon, t_hours))
    n = len(emb)
    if n < 2:
        return []
    cos = emb @ emb.T
    km = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    dt = np.abs(t[:, None] - t[None, :])
    with np.errstate(invalid="ignore"):  # NaN comparisons are False -> pair rejected
        m = (cos >= cos_thr) & (km <= max_km) & (dt <= max_hours)
    m = np.triu(m, 1)
    return [(int(i), int(j), float(cos[i, j]), float(km[i, j]), float(dt[i, j])) for i, j in zip(*np.nonzero(m))]


def to_hours(ts: pd.Series):
    return ((pd.to_datetime(ts) - pd.Timestamp("2000-01-01")).dt.total_seconds() / 3600.0).to_numpy(dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "processed" / "reports_clean.csv"))
    ap.add_argument("--cos", type=float, default=DEFAULTS["cos_thr"])
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    df = pd.read_csv(a.data, keep_default_na=False, parse_dates=["timestamp"])
    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    pairs = find_duplicates(get_embedder().encode(df.text), df.latitude, df.longitude, to_hours(df.timestamp), cos_thr=a.cos)
    print(f"{len(pairs)} duplicate pairs (cos>={a.cos})")
    for i, j, c, km, h in sorted(pairs, key=lambda x: -x[2])[:a.top]:
        print(f"{df.report_id[i]} ~ {df.report_id[j]} cos={c:.3f} km={km:.2f} h={h:.2f}")


if __name__ == "__main__":
    main()
