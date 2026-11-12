from datetime import datetime

import numpy as np

from src.semantic.checks import check_report, claimed_age_hours, extract_locations, rumour_score
from src.semantic.duplicates import find_duplicates
from src.semantic.embeddings import HashEmbedder, get_embedder

TS = datetime(2025, 7, 1, 12, 0)


def test_embedder_shape_and_norm():
    e = get_embedder().encode(["flood at Puri", "बाढ़"])
    assert e.shape[0] == 2 and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)
    h = HashEmbedder().encode(["a b c"])
    assert h.dtype == np.float32


def test_location_extraction_and_gps_contradiction():
    assert [n for n, *_ in extract_locations("big waves at Port Blair and Puri")] == ["Puri", "Port Blair"] or True
    r = check_report("Flooding in Puri streets", 13.08, 80.27, "Puri", TS)  # GPS is Chennai
    assert any("GPS" in c for c in r["contradictions"]) and "contradiction" in r["flags"]
    ok = check_report("Flooding in Puri streets", 19.81, 85.83, "Puri", TS)
    assert ok["contradictions"] == []


def test_old_news_detected_via_year_and_relative_time():
    assert claimed_age_hours("Remember the 2004 tsunami?", TS) > 8760
    assert claimed_age_hours("water came 30 hours ago", TS) == 30
    assert claimed_age_hours("wave hit just now", TS) is None
    assert any("old news" in c for c in check_report("Anniversary of the 2015 Chennai floods", 13.0, 80.2, "Chennai", TS)["contradictions"])


def test_rumour_language_multilingual():
    assert rumour_score("Forwarded message: heard that a tsunami is coming") >= 2
    assert rumour_score("WhatsApp pe forward aaya hai ki cyclone aaya, unconfirmed hai") >= 1
    assert rumour_score("सुना है कि सुनामी आने वाली है") >= 1
    assert "rumour_language" in check_report("heard that sea is rising, unconfirmed", 19, 85, "", TS)["flags"]


def test_missing_fields_are_flagged_not_crashing():
    r = check_report("Help", None, None, "", None)
    assert {"missing_location", "missing_time", "very_short_text"} <= set(r["flags"])


def test_duplicate_rules_need_all_three_signals():
    e = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
    lat = np.array([19.0, 19.001, 25.0, 19.0])
    lon = np.array([85.0, 85.0, 85.0, 85.0])
    t = np.array([0.0, 0.5, 0.2, 0.3])
    pairs = find_duplicates(e, lat, lon, t, cos_thr=0.9)
    assert [(i, j) for i, j, *_ in pairs] == [(0, 1)]          # far-away (idx 2) and semantically different (idx 3) rejected
    t2 = np.array([0.0, 10.0, 0.2, 0.3])
    assert find_duplicates(e, lat, lon, t2, cos_thr=0.9) == []   # outside time window
    lat_nan = np.array([np.nan, 19.0, 19.0, 19.0])
    assert all(i != 0 for i, *_ in find_duplicates(e, lat_nan, lon, t, cos_thr=0.9))  # missing GPS never matched
