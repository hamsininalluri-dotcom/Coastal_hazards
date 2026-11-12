#!/bin/sh
# Idempotent: builds data, trains models, evaluates and seeds the demo DB only when missing.
set -e
[ -f data/processed/reports_clean.csv ] || { python -m src.data.generate_data && python -m src.data.clean_data; }
python -m src.data.validate_data
[ -f models/selected.joblib ] || python -m src.classification.cli train
[ -f reports/metrics.json ] || python -m src.evaluation.evaluate
python -m src.api.seed
