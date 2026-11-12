"""Load the cleaned dataset into the API database (demo data).  python -m src.api.seed [--split test|all]"""
import argparse

import pandas as pd

from src.api.db import make_engine, make_session_factory
from src.api.service import HazardService
from src.pipeline import load_clean


def records_from_df(df):
    out = []
    for r in df.itertuples():
        out.append(dict(report_id=r.report_id, text=r.text, language=r.language,
                        latitude=None if pd.isna(r.latitude) else float(r.latitude), longitude=None if pd.isna(r.longitude) else float(r.longitude),
                        location_text=r.location_text, timestamp=None if pd.isna(r.timestamp) else r.timestamp.to_pydatetime(),
                        source=r.source, media_flag=bool(r.media_flag), true_incident_id=r.true_incident_id))
    return out


def seed(svc, split="test"):
    df = load_clean()
    if split != "all":
        df = df[df.split == split]
    _, warnings = svc.ingest(records_from_df(df), recluster=True)
    return len(df), warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--force", action="store_true", help="seed even if the DB already has reports")
    a = ap.parse_args()
    svc = HazardService(make_session_factory(make_engine()))
    if svc.live_stats()["n_reports"] and not a.force:
        print("database already has data; skipping (use --force)")
        return
    n, w = seed(svc, a.split)
    print(f"seeded {n} reports", w or "")
    print(svc.live_stats())


if __name__ == "__main__":
    main()
