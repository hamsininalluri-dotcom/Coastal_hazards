"""Clean raw reports, flag incomplete ones, and create leakage-free train/val/test splits.

Usage: python -m src.data.clean_data
"""
import argparse
import re
import unicodedata

import numpy as np
import pandas as pd

from src.common import DATA_DIR, INDIA_BBOX, SEED

URL = re.compile(r"https?://\S+")


def clean_text(t) -> str:
    t = unicodedata.normalize("NFKC", str(t) if t is not None and t == t else "")
    t = URL.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"] != ""].drop_duplicates("report_id")
    df["timestamp"] = pd.to_datetime(df["timestamp"].replace("", np.nan), errors="coerce", utc=True).dt.tz_localize(None)
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    la0, la1, lo0, lo1 = INDIA_BBOX
    bad = ~df.latitude.between(la0, la1) | ~df.longitude.between(lo0, lo1)
    df.loc[bad, ["latitude", "longitude"]] = np.nan
    df["location_text"] = df["location_text"].fillna("").map(clean_text)
    df["missing_location"] = df.latitude.isna() | df.longitude.isna()
    df["missing_time"] = df.timestamp.isna()
    df["short_text"] = df.text.str.split().str.len() < 3
    df["needs_triage"] = df.missing_location | df.missing_time
    return df.reset_index(drop=True)


def assign_splits(df: pd.DataFrame, seed=SEED, frac=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    """Group-wise split: every report of one true incident (and its duplicates) lands in ONE split."""
    rng = np.random.RandomState(seed)
    key = np.where(df.true_incident_id.fillna("NONE") != "NONE", df.true_incident_id, df.report_id)
    groups = pd.unique(key)
    rng.shuffle(groups)
    n = len(groups)
    cut1, cut2 = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    gmap = {g: ("train" if i < cut1 else "val" if i < cut2 else "test") for i, g in enumerate(groups)}
    df = df.copy()
    df["split"] = [gmap[k] for k in key]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(DATA_DIR / "raw" / "reports_raw.csv"))
    ap.add_argument("--out", default=str(DATA_DIR / "processed" / "reports_clean.csv"))
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    df = assign_splits(clean_df(pd.read_csv(a.inp, keep_default_na=False)), a.seed)
    df.to_csv(a.out, index=False)
    print(f"cleaned {len(df)} rows -> {a.out}")
    print(df.split.value_counts().to_string())
    print("triage queue size:", int(df.needs_triage.sum()))


if __name__ == "__main__":
    main()
