"""Rule-based keyword classifier: model-failure fallback AND independent re-labeller
used by validate_data.py as a proxy for a second annotator."""
import re

from src.common import ALL_TYPES, HAZARD_TYPES

LEX = {
    "tsunami": ["tsunami", "sunami", "सुनामी", "receded", "peeche gaya", "पीछे हटा", "tremor", "bhukamp", "भूकंप"],
    "storm_surge": ["surge", "sea wall", "embankment", "jwar", "ज्वार", "तटबंध", "tide is abnormally"],
    "flooding": ["flood", "waterlogged", "submerged", "baadh", "बाढ़", "पानी भर", "paani bhar", "dub gay", "डूब"],
    "high_waves": ["high waves", "rough sea", "swells", "unchi lehr", "ऊँची लहरें", "waves over", "rough hai"],
    "coastal_erosion": ["erod", "erosion", "kat raha", "कट रहा", "caving", "dhans", "धँस", "washed away", "foundation exposed", "kaat di"],
    "oil_spill": ["oil", "tar ", "तेल", "slick", "tanker"],
    "cyclone_damage": ["cyclone", "चक्रवात", "roofs", "power lines", "ped aur khambe", "toot"],
}
GUARDS = [r"flood of (emails|orders)", r"wave of (nostalgia|layoffs)", r"tsunami of homework", r"storm on social", r"\blol\b", r"haha",
          r"\b(19|20)\d{2}\b.*\b(remember|anniversary|never forget|throwback|yaad)\b", r"\b(remember|anniversary|throwback)\b",
          r"never forget", r"काम की सुनामी", r"ki baadh aa gayi", r"lehar thi", r"'[a-z]+'", r"calm waves", r"cleanup", r"traffic jam",
          r"beach cleanup", r"fish market", r"lighthouse", r"sunset", r"garmi", r"ice cream", r"सुना है", r"forwarded", r"whatsapp pe forward",
          r"pipe(line)? ?(burst|phat)|phat gayi", r"surfing|surf ", r"tourists", r"petrol|oil prices|queue", r"road repair|repair going",
          r"cricket", r"erosion of trust", r"applause|festival", r"back at .* harbour"]


class KeywordClassifier:
    name = "fallback_keyword"

    def predict(self, texts):
        out = []
        for t in texts:
            tl = (t or "").lower()
            if any(re.search(g, tl) for g in GUARDS):
                scores = {k: 0.0 for k in HAZARD_TYPES}
            else:
                scores = {k: float(sum(w in tl for w in ws)) for k, ws in LEX.items()}
            tot = sum(scores.values())
            if tot == 0:
                probs = {k: 0.02 for k in HAZARD_TYPES}
                probs["not_hazard"] = 0.86
            else:
                probs = {k: 0.8 * v / tot + 0.01 for k, v in scores.items()}
                probs["not_hazard"] = 0.03
            z = sum(probs.values())
            probs = {k: v / z for k, v in probs.items()}
            top = max(HAZARD_TYPES, key=probs.get)
            p_h = 1 - probs["not_hazard"]
            out.append(dict(type=top if p_h >= 0.5 else "not_hazard", type_probs=probs, p_hazard=p_h, relevant=p_h >= 0.5,
                            severity="low" if p_h >= 0.5 else "none", confidence=min(probs[top], 0.6), model_used=self.name))
        return out
