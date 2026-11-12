# Crowdsourced Coastal Hazard Intelligence and Incident Clustering System
Smart India Hackathon 2025 - Problem Statement **SIH25039**

Turns noisy public coastal-hazard reports (tsunami, storm surge, flooding, high waves, coastal erosion, oil spill, cyclone damage) into **structured incident clusters
with confidence and priority**, shown on a response map. Public reports are **never treated as confirmed**: uncertainty is preserved end to end and a human
reviewer can confirm, reject, escalate or ask for more information. Every decision is audit-logged.

## Project outcome
**AI-assisted coastal-hazard intelligence system integrating language classification, semantic reasoning and clustering-based prioritization that transforms noisy public reports into structured incident groups while preserving uncertainty and enabling human response review.**

## What is in the box
| area | what | where |
|---|---|---|
| Data | 2,300-report synthetic multilingual dataset (en / hi / Hinglish), edge + incomplete cases, schema, acceptance criteria | `src/data`, `docs/data_schema.md` |
| Classification | baseline TF-IDF+LR vs selected hybrid (char/word TF-IDF + sentence embeddings, sigmoid-calibrated LR): type, relevance, severity, confidence | `src/classification` |
| Semantic | embeddings, location/time entities, GPS-vs-text and claimed-time contradictions, rumour flags, duplicate pairs | `src/semantic` |
| Clustering | baseline geohash+time window vs DBSCAN/HDBSCAN on spatial+temporal+semantic distance | `src/clustering` |
| Priority Agent | explainable weighted score + guard-rail rules | `src/priority` |
| Evaluation | metrics, baseline comparison, failure analysis, threshold sensitivity, plots | `src/evaluation`, `reports/` |
| API | FastAPI + SQLAlchemy, triage queue, review state machine, audit log | `src/api` |
| Dashboard | Streamlit + Leaflet (folium) map, evidence panel, filters, review form | `frontend/app.py` |
| Ops | Docker, docker-compose, Makefile, pytest | root |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.10+
pip install -r requirements.txt                          # or requirements-lite.txt (no torch; hash-embedder fallback)

make data        # generate -> clean -> VALIDATE (fails if acceptance criteria are not met)
make train       # baseline + selected classifier (fitted on the train split only)
make evaluate    # writes reports/metrics.json, plots, evaluation_report.md
make seed        # load the held-out test reports into the API database (demo data)
make api         # http://localhost:8000/docs
make ui          # http://localhost:8501   (set API_URL if the API is elsewhere)
make test        # pytest -q   (API tests auto-skip if fastapi/sqlalchemy are not installed)
make demo        # scripted end-to-end flow on seeded data
```
**Docker:** `docker compose up --build` → API on :8000, dashboard on :8501. The API container bootstraps data, models, evaluation and demo data on first start.
Use `REQS=requirements.txt HAZARD_EMBEDDER=sbert docker compose up --build` for transformer embeddings.

**Embedder:** `HAZARD_EMBEDDER=auto` (default) tries the multilingual sentence-transformer `paraphrase-multilingual-MiniLM-L12-v2` and silently falls back to a hashed char-n-gram embedder if
`sentence-transformers` or the model download is unavailable. Models remember which embedder they were trained with. The numbers below were produced **with the hash fallback** (no network in the build environment); re-run `make train evaluate` to get transformer numbers.

## Expected output (abridged, seed 42)
`python -m src.pipeline --split test`
```
{ "n_reports": 401, "n_relevant": 328, "n_triage": 13, "n_clusters": 66, "n_multi_report_incidents": 32 }
C0037 [Critical] coastal_erosion reports=25 distinct=18 conf=0.89 (corroborated (still unverified)) status=REPORTED/Unverified
```
(Reports ≠ incidents: 401 reports became 66 clusters, 32 of them supported by ≥ 2 distinct reports; 13 reports went to manual triage.)

## Results (held-out test split, synthetic data)
| | baseline | selected |
|---|---|---|
| type macro-F1 | 0.968 | 0.968 |
| calibration error (ECE) | 0.059 | 0.042 |
| duplicate F1 | 0.795 (exact text) | 0.909 |
| clustering ARI (end-to-end) | 0.657 (geohash+6h) | 0.974 (DBSCAN) |
| clustering NMI / purity | 0.910 / 0.997 | 0.986 / 1.000 |
| clusters vs true incidents | 106 vs 55 | 66 vs 55 |

**Read these honestly.** The data are synthetic and template-based, so absolute scores are optimistic. With the hash fallback the "selected" classifier ties the baseline on accuracy (it is only better calibrated);
the clearest, most trustworthy gains are in clustering and duplicate handling. The one place the classifier genuinely fails is *ambiguous* water reports (error rate 0.63, n=19), which is correct
behaviour to surface: the text does not contain the answer. HDBSCAN with default settings fragments incidents (ARI 0.30) and is offered only as an option.
Full tables, failure analysis, threshold sensitivity and trade-offs: `reports/evaluation_report.md`.

## Making uncertainty visible
- UI banner: *everything is REPORTED, not confirmed*; unverified clusters have dashed outlines; only **Confirmed** gets a solid black outline.
- Confidence is shown with its reasons; clusters below 0.5 (and every single-report cluster) are labelled **unconfirmed**; rumour language and contradictions lower confidence and are listed.
- Priority = urgency *if true*; rule R2 caps low-confidence clusters at Medium.
- Reports with missing location/time go to the **manual triage queue**; model failure falls back to keyword rules with a visible `degraded` flag.

## Limitations
- **Synthetic data**: no real slang, spelling variation, misinformation campaigns or regional languages (Tamil, Bengali, Odia, Malayalam, Telugu, Marathi, Gujarati…). Real performance will be lower; validate on real labelled reports (e.g. with INCOIS / SDMA partners) before any operational use.
- Entity extraction is a gazetteer of 28 place names in Latin script; no Devanagari/regional-script place names, no fuzzy matching, no geocoding of free text.
- Classifier uses embeddings + linear models, not a fine-tuned transformer; ambiguous text is inherently unresolvable.
- Duplicate and cluster thresholds were set by hand and inspected on the test split; re-tune on a validation set.
- Pairwise distance matrices are O(n²): fine for thousands of reports, not millions (use a spatial index / sliding time window at scale).
- Cluster ids are matched across re-clustering by overlap (Jaccard ≥ 0.4); heavy re-grouping can supersede a reviewed cluster (kept, flagged).
- The population-exposure factor is a coarse 3-tier proxy tied to nearby gazetteer towns, not census data.
- Streamlit map click-to-select relies on tooltips; a selector is also provided.

## Ethical considerations
- **Misinformation**: crowdsourced text can be wrong, sarcastic, recycled or deliberately false. The system never auto-confirms, flags rumour/old-news language, shows contradictions, and requires a named human for any status change. Do not use this output for automated public alerts.
- **False negatives vs false positives**: a missed hazard can cost lives, a false alarm erodes trust. Thresholds favour recall for relevance with review as the safety net; the trade-off tables are in the evaluation report.
- **Privacy**: reports carry precise GPS, free text and possibly names/phone numbers (SMS). Collect with consent, store minimally, strip identifiers, restrict reviewer access, set retention limits, and do not link reports to individuals. The demo stores no personal data.
- **Bias**: coverage follows phone and app penetration, language and connectivity - fishing hamlets, elderly and low-income communities report less, so "no reports" is not "no hazard". Multilingual quality will differ by language. Source-credibility weights (app > SMS > social) encode an assumption that can disadvantage some groups; audit them, and treat priority as a triage aid, not a ranking of whose safety matters.
- **Accountability**: the audit log records who changed what and why; keep humans (and official bulletins from IMD/INCOIS) in the loop.

## Future work
Fine-tune MuRIL / XLM-R on real labelled data; add regional-language gazetteers and a geocoder; image/video evidence analysis; cross-check with official feeds (INCOIS, IMD, tide gauges, satellite flood maps); streaming ingestion with incremental clustering and spatial indexes; active learning from reviewer decisions; PostgreSQL/PostGIS; authentication, roles and rate limiting; notification hooks for escalations.

## Structure
```
data/  src/{data,classification,semantic,clustering,priority,evaluation,api}/  frontend/  tests/  reports/  docs/  scripts/
docker-compose.yml  Dockerfile  Makefile  requirements.txt  requirements-lite.txt
```
See `docs/architecture.md` (diagrams, data flow, API table) and `docs/data_schema.md`.
