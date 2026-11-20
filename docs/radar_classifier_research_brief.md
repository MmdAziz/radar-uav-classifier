# Radar UAV / Non-UAV Classifier — Junior Project Research Brief

**Draft date:** 2026-11-13  
**Status:** Planning; dataset and model are not yet selected.  
**Workstream:** Radar-based classification; independent of the existing YOLOv8n vision baseline and the teammate's synthetic radar.

## 1. Scientific objective

Develop and evaluate a reproducible binary classifier that predicts **UAV** versus **non-UAV** from an appropriate, experimentally acquired radar representation (for example, a micro-Doppler spectrogram or FMCW radar-derived image).

This module operates on an already-acquired radar observation. It does **not** independently search an area, estimate 3D coordinates, track targets, provide identification/authorization decisions, or connect to any automated interception logic.

## 2. Current scope and unresolved questions

- Dataset: **to be selected** after a comparison of KTH, DIAT-μSAT, TSMS-Drone and St Andrews/MathWorks.
- Task: first target is **binary classification** (UAV/non-UAV).
- Classes: map source classes only after checking original metadata; document any excluded ambiguous/mixed-target classes.
- Input format: **not fixed yet**. Do not assume every radar dataset consists of JPEG/PNG spectrograms. KTH provides complex-valued FMCW data that requires preprocessing; other candidates provide rendered images.
- Baseline model: small CNN as first option; final architecture depends on actual input and data volume.
- Evaluation: confusion matrix, UAV precision, recall, F1, false-positive rate on non-UAV examples, class balance, calibration/reliability where practical, plus inference time and evaluation protocol.
- Dataset partition: prefer original published train/validation/test splits or recording/target/session-disjoint groups. Do not randomly split correlated frames or neighboring windows without leakage analysis.
- Limitation: benchmark classification performance cannot establish real-world radar range, target localization, or the validity of synthetic radar observations from a coordinate-only simulator.

## 3. Three-day plan

### Friday 2026-11-13 — Research
1. Specify the actual task and non-UAV categories.
2. Research at least four candidate datasets and 5–8 papers.
3. Inspect the repository access, file format, labels, metadata and original evaluation partition for each finalist.
4. Use `radar_dataset_research_catalog.xlsx` as the evidence log.
5. Select one primary dataset and one fallback. Write a decision rationale, not just a preference.

### Saturday 2026-11-14 — Reproducible baseline
1. Set up Python environment/repository and a nonversioned `data/` folder.
2. Inspect actual data and visualize representative samples from each class.
3. Freeze train/validation/test partition before model selection.
4. Create preprocessing, baseline CNN and training routine.
5. Save seeds, configuration, weights and training history.

### Sunday 2026-11-15 — Evaluation and documentation
1. Evaluate the held-out test partition once after model selection.
2. Create confusion matrix, precision, recall, F1 and false-alarm discussion.
3. Compare literature only after accounting for dataset, split, preprocessing, architecture and evaluation differences.
4. Create a standalone inference interface that accepts the same radar representation used for model training.
5. Document known limitations and keep the simulator integration as a clearly labeled future interface.

## 4. Proposed project structure

```text
radar-classifier/
  README.md
  docs/
    radar_classifier_research_brief.md
    dataset_selection.md
    literature_notes.md
  configs/
    baseline.yaml
  src/
    data.py
    model.py
    train.py
    evaluate.py
    predict.py
  tests/
    test_data.py
    test_predict.py
  notebooks/
    01_dataset_exploration.ipynb
  data/              # excluded from Git
  outputs/           # exported experiments and weights, generally excluded from Git
  requirements.txt
  .gitignore
```

Keep the original `radar_design_spec.md` from your teammate under a separate `docs/` path as a **whole-system proposal**. It is not an instruction to automatically implement its synthetic radar, tracker or incorrect/unconfirmed dataset class mapping in this classifier workstream.

## 5. Interface contract (only after dataset selection)

```python
# Example conceptual interface, not yet implemented.
classify_radar_observation(observation) -> {
    "label": "uav" | "non_uav",
    "p_uav": float
}
```

Document `observation` precisely once the dataset is selected (image dimensions, scaling, signal transformation, channels, sample rate, whether radar metadata are required). Do not invent a mapping between complex radar measurements and JPG spectrograms.

## 6. Data integrity checklist

- Dataset and paper DOI/URL recorded.
- Usage license and download path checked.
- Exact labels and class counts inspected.
- Data preparation steps versioned.
- File duplication, adjacent recording overlap and other leakage risks considered.
- One group split/published split fixed before training.
- Positive class explicitly defined as UAV.
- Test set not used for tuning hyperparameters.
- Source publication metrics cited with their original setup, never presented as directly comparable unless protocols match.

## 7. Codex instruction for first coding session

> Read `docs/radar_classifier_research_brief.md` and the dataset-selection document before writing any code. The dataset and its source representation are not finalized. Do not assume images exist or fabricate data. First propose the smallest reproducible project skeleton and a checklist for inspecting the actual downloaded dataset. Do not implement the synthetic radar, target tracking, weapon/target assignment, or simulation control. Wait for the selected dataset's verified metadata before implementing preprocessing and model training.

## 8. Research starting points

- KTH radar dataset: https://zenodo.org/records/5896641
- KTH publication: https://doi.org/10.1109/RadarConf2248738.2022.9764194
- DIAT-μSAT dataset: https://doi.org/10.21227/1x2q-8v62
- DIAT validation pitfalls: https://doi.org/10.3390/s26195989
- TSMS-Drone dataset: https://doi.org/10.25452/figshare.plus.30027313
- TSMS-Drone publication: https://doi.org/10.1038/s41597-026-06802-6
- St Andrews MathWorks image dataset: https://zenodo.org/records/18553708
- St Andrews repository: https://github.com/ZhongliangGuo/MathWorks-Radar-Drone-Project
- Drone/bird micro-Doppler CNN research: https://doi.org/10.1049/iet-rsn.2019.0493
- Micro-Doppler radar signal simulation context (future scope): https://doi.org/10.1049/rsn2.12494
