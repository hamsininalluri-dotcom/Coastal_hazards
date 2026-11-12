"""Priority Agent: transparent weighted score + guard-rail rules -> Critical/High/Medium/Low with explanation.
Priority expresses response urgency IF the reports are true; it never implies confirmation."""
import math
from datetime import datetime

from src.common import nearest_place

WEIGHTS = dict(severity=0.30, volume=0.20, confidence=0.20, source_credibility=0.10, recency=0.10, population_exposure=0.10)
LEVELS = [(0.70, "Critical"), (0.50, "High"), (0.30, "Medium"), (0.0, "Low")]
ORDER = ["Low", "Medium", "High", "Critical"]


def _level(score):
    return next(name for thr, name in LEVELS if score >= thr)


def assess(cluster: dict, now: datetime | None = None) -> dict:
    now = now or datetime.utcnow()
    last = datetime.fromisoformat(cluster["last_report_time"])
    hours = max(0.0, (now - last).total_seconds() / 3600)
    _, km, tier = nearest_place(cluster["centroid_lat"], cluster["centroid_lon"])
    exposure = tier / 3 if km <= 60 else 1 / 3
    raw = dict(severity=cluster["severity_score"], volume=min(1.0, math.log2(1 + cluster["n_unique_reports"]) / math.log2(11)),
               confidence=cluster["confidence"], source_credibility=cluster["source_credibility"],
               recency=math.exp(-hours / 24), population_exposure=exposure)
    factors = [dict(factor=k, value=round(v, 3), weight=WEIGHTS[k], contribution=round(v * WEIGHTS[k], 3)) for k, v in raw.items()]
    score = sum(f["contribution"] for f in factors)
    level, rules = _level(score), []
    if cluster["dominant_type"] == "tsunami" and cluster["n_unique_reports"] >= 3 and cluster["confidence"] >= 0.5 and ORDER.index(level) < 2:
        level = "High"
        rules.append("R1 tsunami with >=3 distinct, reasonably confident reports -> at least High")
    if cluster["confidence"] < 0.35 and ORDER.index(level) > 1:
        level = "Medium"
        rules.append("R2 confidence < 0.35 -> capped at Medium (avoid alarming on weak evidence)")
    if cluster["n_unique_reports"] < 2 and ORDER.index(level) > 1:
        level = "Medium"
        rules.append("R3 single uncorroborated report -> capped at Medium")
    top = sorted(factors, key=lambda f: -f["contribution"])[:3]
    expl = (f"{level} (score {score:.2f}). Main drivers: " + "; ".join(f"{f['factor']}={f['value']:.2f} (+{f['contribution']:.2f})" for f in top)
            + (". Rules: " + " | ".join(rules) if rules else "") + ". Based on UNVERIFIED public reports.")
    return dict(priority=level, score=round(score, 3), factors=factors, rules_applied=rules, explanation=expl)
