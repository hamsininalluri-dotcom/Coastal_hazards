PY ?= python
.PHONY: setup data validate train evaluate seed api ui test demo all clean
setup:     ; pip install -r requirements.txt
data:      ; $(PY) -m src.data.generate_data && $(PY) -m src.data.clean_data && $(PY) -m src.data.validate_data
validate:  ; $(PY) -m src.data.validate_data
train:     ; $(PY) -m src.classification.cli train
evaluate:  ; $(PY) -m src.evaluation.evaluate
seed:      ; $(PY) -m src.api.seed --force
api:       ; uvicorn src.api.main:app --reload --port 8000
ui:        ; streamlit run frontend/app.py
test:      ; $(PY) -m pytest -q
demo:      ; $(PY) scripts/demo.py
all: data train evaluate test
clean:     ; rm -rf data/raw/*.csv data/processed/*.csv data/hazard.db models/*.joblib reports/*.png reports/metrics.json reports/evaluation_report.md
