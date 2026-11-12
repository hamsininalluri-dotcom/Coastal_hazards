"""Check the dataset against the acceptance criteria in docs/data_schema.md. Exit code 1 on failure.

Usage: python -m src.data.validate_data [--path data/processed/reports_clean.csv]
"""
import argparse
import sys

import numpy as np
import pandas as pd

from src.classification.fallback import KeywordClassifier
from src.common import ALL_TYPES, DATA_DIR, INDIA_BBOX, SEVERITIES

REQUIRED = ["report_id", "text", "language", "latitude", "longitude", "location_text", "timestamp", "incident_type", "severity",
            "source", "media_flag", "confidence", "true_incident_id", "is_duplicate"]
EDGE_TAGS = ["sarcasm", "rumour", "old_news", "metaphor", "unrelated", "conflicting_location", "missing_location",
             "missing_time", "short_text", "typo", "code_mixed"]
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == v).mean() * (b == v).mean() for v in np.unique(np.r_[a, b]))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def run_checks(df: pd.DataFrame):
    res = []

    def chk(name, ok, detail):
        res.append((name, bool(ok), detail))

    miss = [c for c in REQUIRED if c not in df.columns]
    chk("A1 required columns present", not miss, f"missing={miss}")
    if miss:
        return res
    chk("A2 at least 2000 reports", len(df) >= 2000, f"n={len(df)}")
    vc = df.incident_type.value_counts(normalize=True)
    chk("A3 all 8 classes present, each >= 4% of rows, max/min <= 6",
        set(vc.index) == set(ALL_TYPES) and vc.min() >= 0.04 and vc.max() / vc.min() <= 6, vc.round(3).to_dict())
    chk("A4 valid severity labels, none <=> not hazard",
        df.severity.isin(SEVERITIES).all() and ((df.severity == "none") == (df.incident_type == "not_hazard")).mean() > 0.97,
        df.severity.value_counts().to_dict())
    mloc = (df.latitude.isna() | df.longitude.isna()).mean()
    mt = df.timestamp.isna().mean() if str(df.timestamp.dtype).startswith("datetime") else (df.timestamp.astype(str).isin(["", "nan", "NaT"])).mean()
    chk("A5 missing location <= 8%, missing time <= 8%, empty text = 0", mloc <= 0.08 and mt <= 0.08 and (df.text.str.len() == 0).sum() == 0,
        f"loc={mloc:.3f} time={mt:.3f}")
    la0, la1, lo0, lo1 = INDIA_BBOX
    ok = df.latitude.dropna().between(la0, la1).all() and df.longitude.dropna().between(lo0, lo1).all()
    chk("A6 coordinates inside India bounding box", ok, "")
    dr = df.is_duplicate.mean()
    orig_ok = df[df.is_duplicate == 1].duplicate_of.isin(df.report_id).all()
    chk("A7 duplicate rate 3-20% and every duplicate_of exists", 0.03 <= dr <= 0.20 and orig_ok, f"rate={dr:.3f}")
    cnt = df.edge_case.value_counts()
    low = {t: int(cnt.get(t, 0)) for t in EDGE_TAGS if cnt.get(t, 0) < 20}
    chk("A8 each edge/incomplete case has >= 20 examples", not low, f"below_threshold={low}")
    lang = df.language.value_counts()
    chk("A9 languages en/hi/hx each >= 10%", all(lang.get(l, 0) / len(df) >= 0.10 for l in ["en", "hi", "hx"]), lang.to_dict())
    kw = KeywordClassifier().predict(df.text.tolist())
    kappa = cohen_kappa((df.incident_type != "not_hazard").astype(int), [int(p["relevant"]) for p in kw])
    chk("A10 label agreement (Cohen kappa vs independent rule labeller on hazard/not) >= 0.60", kappa >= 0.60, f"kappa={kappa:.3f}")
    if "split" in df.columns:
        key = np.where(df.true_incident_id != "NONE", df.true_incident_id, df.report_id)
        leak = pd.Series(df.split.values).groupby(key).nunique().gt(1).sum()
        sizes = df.split.value_counts(normalize=True)
        chk("A11 no incident appears in more than one split", leak == 0, f"leaking_groups={int(leak)}")
        chk("A12 split sizes within +-8pp of 70/15/15", all(abs(sizes.get(k, 0) - v) <= 0.08 for k, v in TARGET.items()), sizes.round(3).to_dict())
        tr = df[df.split == "train"].incident_type.unique()
        chk("A13 every class present in train", set(ALL_TYPES) <= set(tr), "")
    else:
        chk("A11 split column present", False, "run clean_data first")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(DATA_DIR / "processed" / "reports_clean.csv"))
    a = ap.parse_args()
    df = pd.read_csv(a.path, keep_default_na=False, parse_dates=["timestamp"])
    df["latitude"] = pd.to_numeric(df.latitude, errors="coerce")
    df["longitude"] = pd.to_numeric(df.longitude, errors="coerce")
    df["timestamp"] = pd.to_datetime(df.timestamp, errors="coerce")
    res = run_checks(df)
    for n, ok, d in res:
        print(f"[{'PASS' if ok else 'FAIL'}] {n}  {d}")
    bad = [n for n, ok, _ in res if not ok]
    print(f"\n{len(res) - len(bad)}/{len(res)} checks passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
