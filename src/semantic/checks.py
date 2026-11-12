"""Entity extraction + consistency checks + rumour detection (rule/gazetteer based, multilingual-friendly)."""
import re
from datetime import datetime

import numpy as np

from src.common import PLACES, haversine_km

RUMOUR_TERMS = ["forwarded", "heard that", "rumour", "rumor", "unconfirmed", "not sure if", "someone said", "whatsapp pe forward",
                "forward aaya", "सुना है", "अफवाह", "afwah", "pata nahi sach", "पता नहीं सच", "is it true", "fake?"]
_PLACE_RE = {n: re.compile(r"(?<!\w)" + re.escape(n) + r"(?!\w)", re.I) for n in PLACES}
_HOURS = re.compile(r"(\d+)\s*(?:hours?|hrs?|ghante|ghanta|घंटे)\s*(?:ago|pehle|पहले|back)", re.I)
_YEARS_AGO = re.compile(r"(\d+)\s*(?:years?|saal|साल)\s*(?:ago|pehle|पहले)", re.I)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
FAR_KM = 150.0
OLD_HOURS = 12.0


def extract_locations(text):
    return [(n, *PLACES[n][:2]) for n, rx in _PLACE_RE.items() if rx.search(text or "")]


def claimed_age_hours(text, ref: datetime | None):
    """Largest time-in-the-past the text claims (hours), or None."""
    t, found = text or "", []
    found += [float(m) for m in _HOURS.findall(t)]
    if re.search(r"\b(yesterday|last night|kal)\b|कल", t, re.I):
        found.append(24.0)
    for unit, h in (("week", 168), ("month", 720), ("year", 8760)):
        if re.search(rf"\blast {unit}\b", t, re.I):
            found.append(float(h))
    found += [float(y) * 8760 for y in _YEARS_AGO.findall(t)]
    ref_year = ref.year if ref is not None else datetime.utcnow().year
    found += [(ref_year - int(y)) * 8760.0 for y in _YEAR.findall(t) if int(y) < ref_year]
    return max(found) if found else None


def rumour_score(text):
    tl = (text or "").lower()
    return sum(term in tl for term in RUMOUR_TERMS)


def check_report(text, lat=None, lon=None, location_text="", timestamp=None):
    """Return dict: text_locations, gps_text_km, claimed_age_h, rumour_score, contradictions[], flags[]."""
    has_gps = lat is not None and lon is not None and not (np.isnan(lat) or np.isnan(lon))
    ts = timestamp if timestamp is not None and timestamp == timestamp else None
    locs = extract_locations(text)
    contradictions, flags = [], []
    gps_km = None
    if has_gps and locs:
        gps_km = float(min(haversine_km(lat, lon, a, b) for _, a, b in locs))
        if gps_km > FAR_KM:
            contradictions.append(f"text mentions {locs[0][0]} but GPS is {gps_km:.0f} km away")
    if has_gps and location_text in PLACES:
        d = float(haversine_km(lat, lon, *PLACES[location_text][:2]))
        if d > FAR_KM:
            contradictions.append(f"location field '{location_text}' is {d:.0f} km from GPS")
    if len(locs) > 1 and max(float(haversine_km(a[1], a[2], b[1], b[2])) for a in locs for b in locs) > FAR_KM:
        contradictions.append("text mentions several distant places")
    age = claimed_age_hours(text, ts)
    if age is not None and age > OLD_HOURS:
        contradictions.append(f"text refers to events ~{age / 24:.0f} days ago (old news / not current)")
    rs = rumour_score(text)
    if rs:
        flags.append("rumour_language")
    if len((text or "").split()) < 3:
        flags.append("very_short_text")
    if not has_gps:
        flags.append("missing_location")
    if ts is None:
        flags.append("missing_time")
    return dict(text_locations=[n for n, _, _ in locs], gps_text_km=gps_km, claimed_age_h=age, rumour_score=rs,
                contradictions=contradictions, flags=flags + (["contradiction"] if contradictions else []))
