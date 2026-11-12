# Data schema, assumptions, labelling guidelines and acceptance criteria

Files: `data/raw/reports_raw.csv` (generator output) → `data/processed/reports_clean.csv` (cleaned + flags + `split`).
Reproducible: `python -m src.data.generate_data --seed 42` (also `HAZARD_SEED`). The data are **synthetic**: templates in English, Devanagari Hindi
and romanised Hindi ("Hinglish") over a gazetteer of 28 Indian coastal places; no real user content is used.

## Attributes
| column | type | meaning |
|---|---|---|
| report_id | str | unique id `R00001` |
| text | str | report text (cleaned: NFKC, URLs removed, whitespace collapsed) |
| language | `en`/`hi`/`hx` | English / Devanagari Hindi / code-mixed Hinglish (romanised) |
| latitude, longitude | float / empty | reporter GPS, WGS84; empty = missing. Outside the India bbox (5-38N, 66-100E) ⇒ treated as missing |
| location_text | str | place name the reporter typed (may conflict with GPS - deliberate edge case) |
| timestamp | ISO-8601 UTC / empty | time the report was **sent** (not necessarily when the event happened); empty = missing |
| incident_type | enum | `tsunami, storm_surge, flooding, high_waves, coastal_erosion, oil_spill, cyclone_damage, not_hazard` |
| severity | enum | `none` (only for not_hazard), `low`, `moderate`, `high` |
| source | enum | `social`, `app`, `sms` |
| media_flag | 0/1 | a photo/video was attached |
| confidence | 0-1 | annotator confidence in the label (lower for edge cases). Not the model confidence |
| true_incident_id | str | ground-truth incident `INC0001…`; `NONE` for noise, metaphors, rumours (no real event). **Evaluation only - never a model feature** |
| is_duplicate | 0/1 | re-post/forward of an earlier report |
| duplicate_of | str | report_id of the original (for duplicates) |
| dup_group | str | id shared by an original and all its re-posts (ground-truth duplicate pairs) |
| edge_case | enum | `normal, code_mixed, sarcasm, rumour, old_news, metaphor, unrelated, hard_negative, ambiguous, conflicting_location, missing_location, missing_time, short_text, typo` |
| is_hazard | 0/1 | 1 if the text reports a hazard (incl. rumours), 0 for non-hazards |
| *added by clean_data* | | `missing_location, missing_time, short_text, needs_triage, split` |

## Assumptions
1. Real events cluster in space (σ≈1-5 km around a gazetteer place), time (bursty: ~60% of reports within ~1 h of onset, long tail of hours) and meaning.
2. A report is *about* an event if its text asserts a current hazard; metaphors ("flood of emails"), sarcasm, old news ("2004 tsunami"), unrelated beach talk and hard negatives (pipe burst, surfing contest) are `not_hazard`.
3. Rumours ("heard that… unconfirmed") are labelled with the hazard they claim (`is_hazard=1`) but have `true_incident_id=NONE`: they are *not* evidence of a real incident. The semantic module flags them; they are not hidden.
4. `ambiguous` reports (e.g. "water is coming into X") are genuinely type-ambiguous among tsunami/storm surge/flooding/high waves; the label is the true event type, which the text alone cannot reveal. Irreducible error here is expected and desired.
5. Conflicting-location reports keep the **true** incident link; the *text/location field* names a far-away place, GPS is correct.
6. ~15% of reports are re-posts of an earlier one (same/slightly edited text, ≤200 m, ≤45 min later).
7. Synthetic data cannot capture real slang, real misinformation campaigns, or regional-language diversity; see Limitations in the README.

## Labelling guidelines (for replacing the synthetic labels by human ones)
- *Hazard vs not*: label `hazard` only if the author states or strongly implies a hazard happening/ongoing at a place now. Metaphor, joke, sarcasm (quotes around the hazard word, "lol"), anniversary/throwback posts, forecasts of other people, and generic beach chatter are `not_hazard`.
- *Type*: choose the most specific physical phenomenon described (receding sea + big wave ⇒ tsunami; seawater over seawall during a cyclone ⇒ storm_surge; rain/tide inundation ⇒ flooding). If two fit equally, pick the one a coastal-disaster officer would check first and set annotator confidence ≤ 0.5.
- *Severity*: `low` no damage; `moderate` property/traffic impact; `high` people trapped / severe damage / rescue requested.
- *Rumour*: relayed claim without first-hand observation ("forwarded", "heard that", "सुना है") ⇒ keep the claimed type, mark `rumour`.
- *Incident linking*: reports belong to the same incident if they describe the same physical event (same area ≲ 15 km, overlapping time window, same phenomenon). Two different phenomena at one place on different days are different incidents.
- *Double annotation*: at least 10% of items independently labelled by two annotators; Cohen's κ ≥ 0.6 on hazard/not-hazard before the data are used.

## Acceptance criteria (enforced by `python -m src.data.validate_data`, exit code 1 on failure)
| id | criterion |
|---|---|
| A1 | all required columns present |
| A2 | ≥ 2,000 reports |
| A3 | all 8 classes present; each ≥ 4% of rows; max/min class ratio ≤ 6 |
| A4 | valid severity labels; `none` ⇔ `not_hazard` (≥97%) |
| A5 | missing location ≤ 8%, missing time ≤ 8%, no empty text |
| A6 | all coordinates inside the India bounding box |
| A7 | duplicate rate 3-20% and every `duplicate_of` exists |
| A8 | every edge/incomplete case (sarcasm, rumour, old_news, metaphor, unrelated, conflicting_location, missing_location, missing_time, short_text, typo, code_mixed) has ≥ 20 examples |
| A9 | each language `en`, `hi`, `hx` ≥ 10% |
| A10 | label agreement: Cohen's κ ≥ 0.60 between the labels and an independent rule-based labeller on hazard/not-hazard. **This is a smoke-test proxy for double annotation** (the rule labeller is hand-built and imperfect), not a substitute for it |
| A11 | no incident (and no report outside an incident) appears in more than one split |
| A12 | split sizes within ±8 percentage points of 70/15/15 |
| A13 | every class present in the training split |

## Train / validation / test rule
Split unit = `true_incident_id` (unlinked reports are their own unit). All reports of one incident, including duplicates, go to the same split, so duplicates and
near-identical burst reports can never leak from train into test. Models are fitted on `train` only; thresholds are inspected on `val`/`test` and the report states which.
Reports with missing location/time stay in the classification data (text is still informative) but never enter clustering.
