# Preliminary Radar Dataset Audit — St Andrews/MathWorks

**Date:** 2026-11-15  
**Scope:** `train_binary.csv`, `eval_binary.csv`, and `README.txt` supplied by the project researcher. This is a **CSV metadata-only audit**; the full extracted image directory was not available to the auditor. No training has been performed.

## 1. Dataset structure

CSV columns in both splits: `filename`, `drone_1_else_0`, `drone_model`, `radar_freq`, `interval`, `wd`.

The dataset `filename` paths appear relative to `<drone_project>/data/`, e.g. `24GHz/drone/interval_0.25_wd_256/<recording_timestamp>/0.png`. The researcher also confirmed folders `data/24GHz`, `data/94GHz`, `data/207GHz`, and separate `simulation/` data. The downloaded README identifies `train_binary.csv`/`eval_binary.csv` as intended for the binary and drone-model tasks; the `4cls` CSVs are intended for four-class classification.

## 2. Counts and integrity checks

| Metric | Training | Evaluation |
|---|---:|---:|
| Total rows | 23,504 | 2,146 |
| UAV label (`1`) | 11,752 | 917 |
| Non-UAV label (`0`) | 11,752 | 1,229 |
| 24 GHz rows | 6,380 | 519 |
| 94 GHz rows | 9,608 | 779 |
| 207 GHz rows | 7,516 | 848 |
| Distinct source-folder paths | 346 | 21 |
| Missing filenames | 0 | 0 |
| Missing class labels | 0 | 0 |
| Duplicate filenames within split | 0 | 0 |

- The `drone_model` column is empty for all non-UAV rows; this appears intentional but should be verified against the source documentation.
- All rows specify `interval = 0.25` and `wd = 256`.
- There is **no overlap in exact image paths** between training and evaluation and **no overlap in full source-folder paths** (parent directory of each image).
- `train_binary.csv` is balanced. `eval_binary.csv` is not balanced, which is acceptable as long as evaluation metrics and distributions are reported explicitly.

## 3. Evaluation caveats and unresolved checks

1. **Nearby source recordings:** 15 of 21 evaluation source-folder groups have a training source-folder group of the same class and radar frequency whose directory timestamps differ by at most 30 seconds. This is a **screening observation, not proof of leakage**. It is unknown whether these are independent trials or adjacent segments of a longer acquisition. The original trial identifiers and recording boundaries must be checked.
2. **Class/date confounding:** The path timestamps show all non-UAV rows dated `2024-02-06` and all UAV rows dated `2024-03-01` across these two CSV files. Class labels therefore coincide with acquisition date. This is a possible shortcut-learning/generalization concern and does **not** by itself demonstrate leakage.
3. **File integrity not yet audited:** Verify that every CSV path resolves under `<drone_project>/data/`, images are readable, and their dimensions/modes are consistent. This needs to run on the user's Windows system containing the full image dataset.
4. **Validation/test protocol:** The source provides `train_binary.csv` and `eval_binary.csv`. Determine how the reference code uses the evaluation set for checkpoint selection. For an independently held-out test, derive a trial-grouped validation set **from training data only** and reserve `eval_binary.csv` for one final evaluation. State that this is an adapted rather than exact reference protocol when comparing results.
5. **Non-UAV composition:** Verify what the `non-drone/` folder represents: birds only versus birds/clutter/noise or combined data. The binary CSV itself supplies no finer-grained non-UAV label.
6. **Cross-sensor recording correspondence:** Recordings from multiple radar frequencies might be temporally aligned. Use metadata to determine whether cross-sensor observations from the same real-world flight were separated between splits.

## 4. Decision / next action

**Status: Preliminary CSV metadata checks passed; full dataset audit pending.**

Run a read-only local audit on the full extracted directory before training. Save its outputs (counts, missing/corrupt files, image dimensions, group overlap, cross-sensor temporal correspondence, and provenance) under `outputs/audits/`; do not overwrite the original CSVs or image files.

**Research classification:** Binary UAV versus non-UAV using experimentally acquired radar-derived spectrogram images. Synthetic `simulation/` data remains outside the initial baseline.
