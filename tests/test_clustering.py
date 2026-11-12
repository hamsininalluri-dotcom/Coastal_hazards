import numpy as np
import pandas as pd

from src.classification.fallback import KeywordClassifier
from src.clustering.cluster import baseline_cluster, build_clusters, cluster_reports, geohash
from src.common import HAZARD_TYPES
from src.pipeline import cluster_df, enrich, probs_matrix
from src.semantic.embeddings import get_embedder


def test_geohash_reference_value():
    assert geohash(42.6, -5.6, 5) == "ezs42"


def test_baseline_groups_by_cell_and_window():
    lab = baseline_cluster([19.0, 19.001, 19.0], [85.0, 85.001, 85.0], [0.0, 1.0, 30.0])
    assert lab[0] == lab[1] != lab[2]


def _reports():
    rows = []
    txt = {"A": "Sea water entering streets, flooding at Puri, roads submerged", "B": "Cyclone winds tore roofs at Chennai, trees down"}
    spec = [("A", 19.81, 85.83, 0.0)] * 3 + [("B", 13.08, 80.27, 2.0)] * 3 + [("A", 19.81, 85.83, 200.0)]
    for i, (k, la, lo, h) in enumerate(spec):
        rows.append(dict(report_id=f"X{i}", text=txt[k] + (" " * i), latitude=la + i * 1e-4, longitude=lo, location_text="",
                         timestamp=pd.Timestamp("2025-07-01") + pd.Timedelta(hours=h + i * 0.1), source="app", media_flag=i % 2, language="en"))
    return pd.DataFrame(rows)


def test_dbscan_separates_place_type_and_time_and_never_drops_reports():
    e = enrich(_reports(), KeywordClassifier(), get_embedder())
    labels, clusters, pairs = cluster_df(e)
    assert len(labels) == 7
    assert labels[0] == labels[1] == labels[2] and labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3] and labels[6] not in (labels[0], labels[3])  # same place, 200 h later -> different incident
    assert sum(c["n_reports"] for c in clusters) == 7


def test_cluster_summary_separates_reports_from_distinct_incidents():
    e = enrich(_reports(), KeywordClassifier(), get_embedder())
    _, clusters, pairs = cluster_df(e)
    big = max(clusters, key=lambda c: c["n_reports"])
    for k in ["cluster_id", "report_ids", "n_reports", "n_unique_reports", "duplicate_count", "centroid_lat", "centroid_lon", "spread_km",
              "first_report_time", "last_report_time", "dominant_type", "confidence", "confidence_label", "confidence_reasons"]:
        assert k in big
    assert big["n_reports"] == 3 and big["n_unique_reports"] + big["duplicate_count"] == 3
    assert big["spread_km"] < 1 and big["dominant_type"] in HAZARD_TYPES
    singles = [c for c in clusters if c["n_reports"] == 1]
    assert singles and singles[0]["confidence_label"] == "unconfirmed"  # lone report stays unconfirmed


def test_duplicate_pairs_collapse_in_unique_count():
    e = enrich(_reports().iloc[:3], KeywordClassifier(), get_embedder())
    e["text"] = e.text.iloc[0]
    labels = np.zeros(3, dtype=int)
    c = build_clusters(e, labels, [(0, 1)])[0]
    assert c["n_reports"] == 3 and c["n_unique_reports"] == 2 and c["duplicate_count"] == 1


def test_empty_and_single_inputs_are_safe():
    assert len(cluster_reports(np.zeros((0, 4)), np.zeros((0, 7)), np.array([]), np.array([]), np.array([]))) == 0
    assert list(cluster_reports(np.ones((1, 4)), np.ones((1, 7)) / 7, np.array([19.0]), np.array([85.0]), np.array([0.0]))) == [0]
