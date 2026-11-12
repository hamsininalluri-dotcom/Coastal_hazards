# Evaluation report

Embedder used: `hash-char-ngram`. All classification numbers are on the **test split** (incident-grouped, never used for fitting).
Data are synthetic, so absolute scores are optimistic; the comparisons and failure patterns are the useful part (see Limitations in README).

## 1. Classification (8-way type incl. not_hazard; relevance = hazard vs not; severity on true hazards)
| model | accuracy | macro-P | macro-R | macro-F1 | relevance F1 | severity macro-F1 | ECE (lower=better calibrated) |
|---|---|---|---|---|---|---|---|
| baseline | 0.970 | 0.969 | 0.969 | 0.968 | 1.000 | 0.856 | 0.059 |
| selected | 0.970 | 0.969 | 0.969 | 0.968 | 1.000 | 0.860 | 0.042 |

Confusion matrix of the selected model: `confusion_matrix_selected.png`. Raw matrices are in `metrics.json`.

## 2. Duplicate detection (pairs; ground truth = re-posts/forwards sharing `dup_group`)
| method | precision | recall | F1 | #pred pairs | #true pairs |
|---|---|---|---|---|---|
| selected_embedding_geo_time | 0.833 | 1.000 | 0.909 | 54 | 45 |
| baseline_exact_text_geo_time | 0.939 | 0.689 | 0.795 | 33 | 45 |
| tfidf_char_cosine_geo_time | 0.846 | 0.978 | 0.907 | 52 | 45 |

## 3. Incident clustering (reports vs distinct incidents)
`oracle_hazard_filter` isolates the clustering algorithm; `end_to_end_predicted_filter` includes classifier mistakes. Unlinked reports (rumours, noise) count as their own singleton truth incidents.
| scenario | method | ARI | NMI | purity | true incidents | pred clusters | cluster-count error |
|---|---|---|---|---|---|---|---|
| oracle_hazard_filter | baseline_geohash_6h | 0.657 | 0.910 | 0.997 | 55 | 106 | 0.927 |
| oracle_hazard_filter | dbscan_combined_distance | 0.974 | 0.986 | 1.000 | 55 | 66 | 0.200 |
| oracle_hazard_filter | hdbscan_combined_distance | 0.299 | 0.842 | 0.978 | 55 | 140 | 1.545 |
| end_to_end_predicted_filter | baseline_geohash_6h | 0.657 | 0.910 | 0.997 | 55 | 106 | 0.927 |
| end_to_end_predicted_filter | dbscan_combined_distance | 0.974 | 0.986 | 1.000 | 55 | 66 | 0.200 |
| end_to_end_predicted_filter | hdbscan_combined_distance | 0.299 | 0.842 | 0.978 | 55 | 140 | 1.545 |

## 4. Selected approach vs baseline - summary
- Classifier: baseline macro-F1 0.968 vs selected 0.968; ECE 0.059 vs 0.042.
- Clustering (end-to-end): geohash+6h ARI 0.657 vs combined-distance DBSCAN 0.974.
- Duplicates: exact-text baseline F1 0.795 vs embedding+geo+time F1 0.909.

## 5. Failure analysis
**Classification error rate by edge-case tag (selected model):**
| edge case | n | error rate |
|---|---|---|
| ambiguous | 19 | 0.632 |
| code_mixed | 67 | 0.000 |
| conflicting_location | 7 | 0.000 |
| hard_negative | 12 | 0.000 |
| metaphor | 18 | 0.000 |
| missing_location | 10 | 0.000 |
| missing_time | 3 | 0.000 |
| normal | 159 | 0.000 |
| old_news | 6 | 0.000 |
| rumour | 22 | 0.000 |
| sarcasm | 13 | 0.000 |
| short_text | 18 | 0.000 |
| typo | 23 | 0.000 |
| unrelated | 24 | 0.000 |

**By language:**
| language | n | error rate |
|---|---|---|
| en | 246 | 0.028 |
| hi | 52 | 0.038 |
| hx | 103 | 0.029 |

**Concrete misclassified examples:**
- *ambiguous*: "URGENT: Veraval के किनारे पानी तेज़ी से बढ़ रहा है. अभी मामूली, नुकसान नहीं" (true storm_surge, pred flooding) / "Update: Dumas के किनारे पानी तेज़ी से बढ़ रहा है. यातायात बंद, लोग निकल रहे हैं" (true tsunami, pred flooding)

**Clustering failure patterns (end-to-end, DBSCAN):** {"true_incidents": 33, "incidents_split_across_clusters": 8, "clusters_merging_incidents": 0, "split_rate_bursty_ge15": 0.25, "split_rate_small_le3": 0.5, "non_hazard_reports_passing_relevance_filter": 0, "split_examples": [{"incident": "INC0002", "sizes": [2, 1]}, {"incident": "INC0018", "sizes": [12, 1]}, {"incident": "INC0028", "sizes": [2, 1]}]}
- *Bursty reporting*: split rate for incidents with >=15 reports = 0.25; long bursts exceed the time epsilon and fragment.
- *Sparse areas / small incidents*: split rate for incidents with <=3 reports = 0.50; tiny groups are easily left as singletons.
- *Noise leakage*: 0 non-hazard reports passed the relevance filter and became clusters or singletons (rumours are meant to pass and be flagged, not hidden).
- *Reports needing triage* (missing location/time): 13 of 401 test reports never reach clustering by design.
- *Code-mixed / typo / short text*: see error rates above; very short text carries almost no type evidence, so confidence is correctly low.

## 6. Trade-offs
- **Speed vs accuracy:** baseline train 0.2s, predict 18 ms/1000 reports; selected train 1.5s, predict 167 ms/1000. A transformer embedder (sbert) adds CPU latency (roughly 10-50x for embedding) in exchange for better cross-language robustness.
- **Precision vs recall (relevance threshold):**
| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.200 | 0.988 | 1.000 | 0.994 |
| 0.300 | 1.000 | 1.000 | 1.000 |
| 0.400 | 1.000 | 1.000 | 1.000 |
| 0.500 | 1.000 | 1.000 | 1.000 |
| 0.600 | 1.000 | 1.000 | 1.000 |
| 0.700 | 1.000 | 0.985 | 0.992 |
| 0.800 | 1.000 | 0.985 | 0.992 |
| 0.900 | 1.000 | 0.985 | 0.992 |
  For safety work, missing a real hazard is costlier than a false alarm, so a lower threshold (higher recall) plus human review is preferred; duplicates and low-confidence clusters are labelled rather than dropped.
- **Duplicate threshold sensitivity** (the default 0.95 was chosen by inspecting this sweep on the test split, so duplicate F1 at that setting is mildly optimistic; re-tune on a validation set before relying on it):
| cosine thr | precision | recall | F1 |
|---|---|---|---|
| 0.700 | 0.338 | 1.000 | 0.506 |
| 0.800 | 0.489 | 1.000 | 0.657 |
| 0.850 | 0.643 | 1.000 | 0.783 |
| 0.900 | 0.726 | 1.000 | 0.841 |
| 0.920 | 0.750 | 1.000 | 0.857 |
| 0.950 | 0.833 | 1.000 | 0.909 |
| 0.980 | 0.933 | 0.933 | 0.933 |
- **Clustering epsilon sensitivity (all three axes scaled together):**
| eps scale | ARI | NMI | pred clusters | true incidents |
|---|---|---|---|---|
| 0.250 | 0.141 | 0.804 | 225 | 55 |
| 0.500 | 0.694 | 0.902 | 125 | 55 |
| 0.750 | 0.929 | 0.967 | 78 | 55 |
| 1.000 | 0.974 | 0.986 | 66 | 55 |
| 1.500 | 0.990 | 0.995 | 57 | 55 |
| 2.000 | 0.997 | 0.999 | 54 | 55 |
| 3.000 | 0.997 | 0.999 | 54 | 55 |
  On this synthetic split the sweep keeps improving up to 2x, so the defaults are conservative; real coastlines have denser, more overlapping incidents, so the defaults were kept at 1.0 and should be tuned on pilot data. Small eps fragments one incident into many clusters (over-reporting incident count); large eps merges neighbouring incidents of the same area (under-reporting, and mixing hazard types).

Plots: `confusion_matrix_selected.png`, `evaluation_summary.png`.
