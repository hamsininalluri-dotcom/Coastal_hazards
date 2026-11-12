import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.classification.fallback import KeywordClassifier

T = "2025-07-01T10:{:02d}:00"


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(f"sqlite:///{tmp_path}/t.db", classifier=KeywordClassifier())) as c:
        yield c


def post(c, text="Cyclone winds tore roofs off houses in Chennai, trees down", i=0, **kw):
    body = dict(text=text, latitude=13.08 + i * 1e-4, longitude=80.27, timestamp=T.format(i), source="app")
    body.update(kw)
    return c.post("/reports", json=body)


def test_health_and_metrics(client):
    assert client.get("/health").json()["status"] == "ok"
    m = client.get("/metrics").json()
    assert "live" in m and "offline_evaluation" in m


def test_missing_location_or_time_goes_to_triage_queue(client):
    r = post(client, latitude=None, longitude=None)
    assert r.status_code == 201 and r.json()["report"]["queue"] == "triage"
    r2 = post(client, timestamp=None)
    assert r2.json()["report"]["queue"] == "triage"
    assert len(client.get("/reports", params={"triage": True}).json()) == 2
    assert client.get("/clusters").json() == []


def test_empty_text_rejected(client):
    assert client.post("/reports", json={"text": ""}).status_code == 422
    assert client.post("/reports", json={"text": "   http://x.y  "}).status_code == 422


def test_non_hazard_report_is_stored_but_not_clustered(client):
    r = post(client, text="Beautiful sunset at the beach today")
    assert r.json()["report"]["queue"] == "not_relevant"


def test_reports_cluster_and_remain_unverified(client):
    for i in range(3):
        post(client, i=i, text=f"Cyclone winds tore roofs off houses in Chennai, trees down{'!' * i}")
    cl = client.get("/clusters").json()
    assert len(cl) == 1 and cl[0]["n_reports"] == 3 and cl[0]["review_status"] == "Unverified"
    assert "not confirmed" in cl[0]["status_note"]
    d = client.get(f"/clusters/{cl[0]['cluster_id']}").json()
    assert len(d["members"]) == 3 and d["audit_log"] == []


def test_filters(client):
    for i in range(3):
        post(client, i=i)
    assert len(client.get("/clusters", params={"hazard_type": "cyclone_damage"}).json()) == 1
    assert client.get("/clusters", params={"hazard_type": "oil_spill"}).json() == []
    assert client.get("/clusters", params={"status": "Confirmed"}).json() == []
    assert client.get("/clusters", params={"status": "Bogus"}).status_code == 422


def test_review_workflow_and_audit_log(client):
    for i in range(3):
        post(client, i=i)
    cid = client.get("/clusters").json()[0]["cluster_id"]
    bad = client.post(f"/clusters/{cid}/review", json={"state": "Confirmed", "reviewer": "asha"})
    assert bad.status_code == 409                       # cannot skip "Under Review"
    assert client.post(f"/clusters/{cid}/review", json={"state": "Under Review", "reviewer": "asha"}).status_code == 200
    ok = client.post(f"/clusters/{cid}/review", json={"state": "Confirmed", "reviewer": "asha", "notes": "INCOIS bulletin matches"})
    body = ok.json()
    assert body["review_status"] == "Confirmed" and len(body["audit_log"]) == 2
    assert body["audit_log"][-1]["reviewer"] == "asha" and body["audit_log"][-1]["notes"].startswith("INCOIS")


def test_review_state_survives_new_reports_and_unknown_cluster_404(client):
    for i in range(3):
        post(client, i=i)
    cid = client.get("/clusters").json()[0]["cluster_id"]
    client.post(f"/clusters/{cid}/review", json={"state": "Under Review", "reviewer": "r1"})
    post(client, i=5)  # new evidence joins the same incident
    c = client.get(f"/clusters/{cid}").json()
    assert c["review_status"] == "Under Review" and c["n_reports"] == 4 and c["changed_since_review"] is True
    assert client.get("/clusters/C9999").status_code == 404
    assert client.post("/clusters/C9999/review", json={"state": "Under Review", "reviewer": "x"}).status_code == 404
