"""CLI: python -m src.clustering.cli [--split test] [--algo dbscan|baseline|hdbscan]"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.clustering.cluster import baseline_cluster
from src.common import DATA_DIR
from src.pipeline import cluster_df, enrich, load_classifier
from src.semantic.duplicates import to_hours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--algo", default="dbscan", choices=["dbscan", "hdbscan", "baseline"])
    a = ap.parse_args()
    df = pd.read_csv(DATA_DIR / "processed" / "reports_clean.csv", keep_default_na=False, parse_dates=["timestamp"])
    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df.timestamp, errors="coerce")
    df = df[(df.split == a.split) & (df.is_hazard == 1)].dropna(subset=["latitude", "timestamp"])
    e = enrich(df, load_classifier())
    if a.algo == "baseline":
        labels = baseline_cluster(e.latitude, e.longitude, to_hours(e.timestamp))
    else:
        labels, clusters, _ = cluster_df(e, a.algo)
    truth = np.where(e.true_incident_id == "NONE", "u" + e.report_id, e.true_incident_id)
    print(f"{a.algo}: {len(e)} reports -> {len(set(labels))} clusters (true incidents {len(set(truth))})  "
          f"ARI={adjusted_rand_score(truth, labels):.3f} NMI={normalized_mutual_info_score(truth, labels):.3f}")


if __name__ == "__main__":
    main()
