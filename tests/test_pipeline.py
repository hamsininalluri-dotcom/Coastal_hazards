from src.classification.fallback import KeywordClassifier
from src.data.clean_data import assign_splits, clean_df
from src.data.generate_data import gen
from src.pipeline import run


def test_end_to_end_with_fallback_classifier():
    df = assign_splits(clean_df(gen(42, 2100)))
    df = df[df.split == "test"]
    r = run(df, clf=KeywordClassifier())
    s = r["summary"]
    assert s["n_reports"] == len(df) and s["n_triage"] == int(df.needs_triage.sum())
    assert s["n_clusters"] > 0 and r["clusters"]
    assert all(c["review_status"] == "Unverified" and c["priority"] in ("Critical", "High", "Medium", "Low") for c in r["clusters"])
    triaged = set(r["triage"].report_id)
    assert all(not (set(c["report_ids"]) & triaged) for c in r["clusters"])  # incomplete reports never clustered
