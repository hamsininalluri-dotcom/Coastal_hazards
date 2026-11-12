"""Shared constants, gazetteer and geo helpers."""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
SEED = int(os.getenv("HAZARD_SEED", "42"))

HAZARD_TYPES = ["tsunami", "storm_surge", "flooding", "high_waves",
                "coastal_erosion", "oil_spill", "cyclone_damage"]
ALL_TYPES = HAZARD_TYPES + ["not_hazard"]
SEVERITIES = ["none", "low", "moderate", "high"]
SEVERITY_SCORE = {"none": 0.0, "low": 0.33, "moderate": 0.66, "high": 1.0}
SOURCES = ["social", "app", "sms"]
SOURCE_CREDIBILITY = {"app": 0.75, "sms": 0.6, "social": 0.4}

# name: (lat, lon, population-exposure tier 1..3)
PLACES = {
    "Mumbai": (19.0760, 72.8777, 3), "Chennai": (13.0827, 80.2707, 3),
    "Visakhapatnam": (17.6868, 83.2185, 3), "Kochi": (9.9312, 76.2673, 3),
    "Thiruvananthapuram": (8.5241, 76.9366, 3), "Puri": (19.8135, 85.8312, 2),
    "Paradip": (20.3166, 86.6114, 2), "Mangaluru": (12.9141, 74.8560, 2),
    "Panaji": (15.4909, 73.8278, 2), "Dumas": (21.0667, 72.7000, 2),
    "Kandla": (23.0333, 70.2167, 2), "Alappuzha": (9.4981, 76.3388, 2),
    "Kanyakumari": (8.0883, 77.5385, 2), "Nagapattinam": (10.7672, 79.8449, 2),
    "Puducherry": (11.9416, 79.8083, 2), "Kakinada": (16.9891, 82.2475, 2),
    "Port Blair": (11.6234, 92.7265, 2), "Okha": (22.4676, 69.0714, 1),
    "Veraval": (20.9077, 70.3679, 1), "Ratnagiri": (16.9902, 73.3120, 1),
    "Rameswaram": (9.2876, 79.3129, 1), "Machilipatnam": (16.1875, 81.1389, 1),
    "Gopalpur": (19.2583, 84.9000, 1), "Mahabalipuram": (12.6269, 80.1927, 1),
    "Malpe": (13.3500, 74.7000, 1), "Karwar": (14.8137, 74.1297, 1),
    "Digha": (21.6266, 87.5093, 1), "Sagar Island": (21.6500, 88.0500, 1),
}
INDIA_BBOX = (5.0, 38.0, 66.0, 100.0)  # lat_min, lat_max, lon_min, lon_max


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km (NaN in -> NaN out)."""
    lat1, lon1, lat2, lon2 = map(lambda x: np.radians(np.asarray(x, dtype=float)),
                                 (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_place(lat, lon):
    """Return (name, km, tier) of nearest gazetteer place."""
    best = min(PLACES.items(), key=lambda kv: float(haversine_km(lat, lon, kv[1][0], kv[1][1])))
    return best[0], float(haversine_km(lat, lon, best[1][0], best[1][1])), best[1][2]
