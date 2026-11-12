from datetime import datetime

from src.priority.agent import ORDER, WEIGHTS, assess

BASE = dict(dominant_type="flooding", n_unique_reports=8, confidence=0.9, severity_score=1.0, source_credibility=0.7,
            centroid_lat=19.076, centroid_lon=72.877, last_report_time="2025-07-01T10:00:00")
NOW = datetime(2025, 7, 1, 11, 0)


def test_weights_sum_to_one_and_factors_explain_score():
    assert abs(sum(WEIGHTS.values()) - 1) < 1e-9
    a = assess(BASE, NOW)
    assert abs(sum(f["contribution"] for f in a["factors"]) - a["score"]) < 0.01
    assert a["priority"] in ORDER and "UNVERIFIED" in a["explanation"]


def test_strong_cluster_outranks_weak_cluster():
    weak = dict(BASE, n_unique_reports=1, confidence=0.3, severity_score=0.33, source_credibility=0.4)
    assert ORDER.index(assess(BASE, NOW)["priority"]) > ORDER.index(assess(weak, NOW)["priority"])


def test_low_confidence_is_capped_at_medium():
    a = assess(dict(BASE, confidence=0.2), NOW)
    assert a["priority"] in ("Low", "Medium")


def test_tsunami_rule_raises_to_at_least_high():
    c = dict(BASE, dominant_type="tsunami", n_unique_reports=3, confidence=0.55, severity_score=0.33, source_credibility=0.4,
             centroid_lat=21.65, centroid_lon=88.05)
    a = assess(c, NOW)
    assert ORDER.index(a["priority"]) >= ORDER.index("High") and a["rules_applied"]


def test_stale_clusters_lose_recency():
    fresh = assess(BASE, NOW)["score"]
    stale = assess(BASE, datetime(2025, 7, 10))["score"]
    assert stale < fresh
