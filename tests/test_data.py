import numpy as np

from src.data.clean_data import assign_splits, clean_df, clean_text
from src.data.generate_data import gen
from src.data.validate_data import cohen_kappa, run_checks


def _clean(seed=42):
    return assign_splits(clean_df(gen(seed, 2100)))


def test_generation_is_reproducible_and_large_enough():
    a, b = gen(7, 2100), gen(7, 2100)
    assert a.equals(b) and len(a) >= 2000


def test_required_edge_cases_present():
    tags = set(gen(3, 2100).edge_case)
    for t in ["sarcasm", "rumour", "old_news", "metaphor", "typo", "short_text", "missing_location", "missing_time", "conflicting_location", "code_mixed"]:
        assert t in tags


def test_clean_text_removes_urls_and_whitespace():
    assert clean_text("  Flood   at   Puri https://x.co/a  ") == "Flood at Puri"


def test_incomplete_reports_flagged_for_triage():
    df = _clean()
    assert df.needs_triage.sum() > 0
    assert (df.needs_triage == (df.missing_location | df.missing_time)).all()


def test_splits_have_no_incident_leakage():
    df = _clean()
    inc = df[df.true_incident_id != "NONE"]
    assert (inc.groupby("true_incident_id").split.nunique() == 1).all()


def test_validation_passes_on_generated_data():
    bad = [(n, d) for n, ok, d in run_checks(_clean()) if not ok]
    assert not bad, bad


def test_validation_fails_when_a_class_is_missing():
    df = _clean()
    df = df[df.incident_type != "oil_spill"]
    assert any(not ok for _, ok, _ in run_checks(df))


def test_validation_fails_on_leaky_split():
    df = _clean()
    i = df[df.true_incident_id != "NONE"].index[:2]
    df.loc[i[0], "split"], df.loc[i[1], "split"] = "train", "test"
    df.loc[i[1], "true_incident_id"] = df.loc[i[0], "true_incident_id"]
    assert any((not ok) and name.startswith("A11") for name, ok, _ in run_checks(df))


def test_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) <= 0.0
