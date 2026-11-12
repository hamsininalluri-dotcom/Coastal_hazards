"""End-to-end demo with seeded data:  python scripts/demo.py
1) data -> validate -> train  2) pipeline on held-out reports  3) triage + review workflow (needs sqlalchemy)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def sh(*a):
    print("\n$ python -m", " ".join(a))
    subprocess.run([sys.executable, "-m", *a], check=True)


sh("src.data.generate_data", "--seed", "42")
sh("src.data.clean_data")
sh("src.data.validate_data")
if not (ROOT / "models" / "selected.joblib").exists():
    sh("src.classification.cli", "train")

print("\n=== Pipeline on the held-out TEST split: reports -> incidents -> priority ===")
sh("src.pipeline", "--split", "test", "--top", "6")

try:
    from src.api.db import make_engine, make_session_factory
    from src.api.service import HazardService
    from src.api.seed import seed
except ImportError as e:
    print(f"\n(skipping API/review demo: {e.name} not installed - run `pip install -r requirements.txt`)")
    sys.exit(0)

print("\n=== API service demo (temporary SQLite database) ===")
db = Path(tempfile.mkdtemp()) / "demo.db"
svc = HazardService(make_session_factory(make_engine(f"sqlite:///{db}")))
n, warn = seed(svc, "test")
print(f"seeded {n} reports; live stats:", svc.live_stats())
top = svc.list_clusters(min_unique=3)[0]
print(f"\nTop cluster {top['cluster_id']}: {top['dominant_type']} | {top['n_reports']} reports / {top['n_unique_reports']} distinct | "
      f"{top['priority']} | {top['review_status']} ({top['status_note']})")
print("explanation:", top["explanation"])
print("\nA report with no location arrives ->")
rep, _ = svc.ingest([dict(text="Huge waves flooding the harbour road, please help", source="sms")])
print("   queue:", rep[0]["queue"], rep[0]["triage_reasons"])
print("\nHuman review:")
for state, note in [("Under Review", "calling district control room"), ("Escalated", "matches IMD alert; forwarding to SDMA")]:
    d = svc.review(top["cluster_id"], state, "demo_reviewer", note)
    print(f"   -> {d['review_status']}")
try:
    svc.review(top["cluster_id"], "Unverified", "demo_reviewer")
except ValueError as e:
    print("   invalid transition rejected:", e)
print("audit log:", [(a["old_status"], a["new_status"], a["reviewer"]) for a in d["audit_log"]])
