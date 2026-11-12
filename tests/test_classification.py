import numpy as np

from src.classification.fallback import KeywordClassifier
from src.classification.model import HazardClassifier
from src.common import ALL_TYPES
from src.data.clean_data import assign_splits, clean_df
from src.data.generate_data import gen

_DF = assign_splits(clean_df(gen(42, 2100)))
_TR, _TE = _DF[_DF.split == "train"], _DF[_DF.split == "test"]


def test_baseline_learns_and_outputs_valid_probabilities():
    m = HazardClassifier("baseline").fit(_TR.text, _TR.incident_type, _TR.severity)
    pred = m.predict(_TE.text.tolist())
    acc = np.mean([p["type"] == t for p, t in zip(pred, _TE.incident_type)])
    assert acc > 0.85
    assert all(abs(sum(p["type_probs"].values()) - 1) < 1e-6 and set(p["type_probs"]) == set(ALL_TYPES) for p in pred)


def test_selected_model_is_calibrated_and_filters_metaphors():
    m = HazardClassifier("selected").fit(_TR.text, _TR.incident_type, _TR.severity)
    r = m.predict(["Got a flood of emails after the long holiday", "Cyclone winds tore roofs off houses in Chennai, trees down"])
    assert r[0]["relevant"] is False and r[0]["type"] == "not_hazard" and r[0]["severity"] == "none"
    assert r[1]["relevant"] is True and r[1]["type"] == "cyclone_damage"
    assert all(0 <= x["confidence"] <= 1 for x in r)


def test_ambiguous_text_gets_lower_confidence_than_clear_text():
    m = HazardClassifier("selected").fit(_TR.text, _TR.incident_type, _TR.severity)
    clear, vague = m.predict(["Black oil slick washing ashore at Puri, strong smell and dead fish", "water coming near the shore"])
    assert clear["confidence"] > vague["confidence"]


def test_keyword_fallback_works_without_a_trained_model():
    r = KeywordClassifier().predict(["Tsunami waves reported at Kochi", "nice sunset today"])
    assert r[0]["relevant"] and r[0]["type"] == "tsunami" and not r[1]["relevant"]
    assert r[0]["model_used"] == "fallback_keyword" and r[0]["confidence"] <= 0.6


def test_model_save_load_roundtrip_remembers_its_embedder():
    import pathlib, tempfile
    path = pathlib.Path(tempfile.mkdtemp()) / "m.joblib"
    m = HazardClassifier("selected").fit(_TR.text, _TR.incident_type, _TR.severity)
    assert m.embedder_name.startswith("hash")
    m.save(path)
    m2 = HazardClassifier.load("selected", path)
    t = "Oil spill near Kochi harbour, black slick on the beach"
    assert m2.predict([t])[0]["type"] == m.predict([t])[0]["type"] == "oil_spill"
