"""CLI: python -m src.classification.cli train|predict ..."""
import argparse
import json

import pandas as pd

from src.classification.model import HazardClassifier
from src.common import DATA_DIR, MODELS_DIR


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default=str(DATA_DIR / "processed" / "reports_clean.csv"))
    t.add_argument("--kinds", nargs="+", default=["baseline", "selected"])
    p = sub.add_parser("predict")
    p.add_argument("text", nargs="+")
    p.add_argument("--kind", default="selected")
    a = ap.parse_args()
    if a.cmd == "train":
        MODELS_DIR.mkdir(exist_ok=True)
        df = pd.read_csv(a.data, keep_default_na=False)
        tr = df[df.split == "train"]  # val/test never used for fitting
        for k in a.kinds:
            m = HazardClassifier(k).fit(tr.text, tr.incident_type, tr.severity)
            print(f"trained {k} on {len(tr)} reports -> {m.save()}")
    else:
        m = HazardClassifier.load(a.kind)
        for txt, r in zip(a.text, m.predict(a.text)):
            r["type_probs"] = {k: round(v, 3) for k, v in r["type_probs"].items()}
            print(json.dumps({"text": txt, **r}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
