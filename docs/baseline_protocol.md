# Baseline Protocol — Radar UAV / Non-UAV Classifier

**Protocol ID:** RP-001  
**Draft date:** 2026-11-15  
**Status:** CANDIDATE / NOT FROZEN. Freeze only after checking the distributed archive and labels.  
**Reference:** Guo et al., *Machine Learning-based Drone and Bird Classification by Micro-Doppler Radar Signatures*, technical report, and associated repository.

## A. Scientific and engineering boundary

Input is an **already generated micro-Doppler image**. Output is a per-observation class label and probability; the model does not determine the existence, direction, or 3D location of targets. The team member's coordinate-based synthetic radar does not automatically produce compatible spectrograms. Do not claim end-to-end radar/simulator validation.

## B. Planned baseline (reference-aligned settings)

| Parameter | Proposed value | Basis / caveat |
|---|---|---|
| Dataset | St Andrews processed images | Pending archive inspection |
| Classification task | Binary, UAV positive; non-UAV negative | Authors' `binary` task; verify label mapping |
| Source representation | 512×512 spectrogram images | Source report, pp. 8–9 |
| Input transformation | PIL RGB; `Resize((224,224))`; `ToTensor()` | GitHub `dataset.py`; no arbitrary image augmentation initially |
| Architecture | ResNet18 | Authors' binary baseline |
| Initialization | ImageNet-pretrained | Reference comparison 'with PT' |
| Loss / optimizer | CrossEntropyLoss / Adam | GitHub `train.py` |
| Learning rate | 1e-5 | Report and repository |
| Batch size | 64 | Report and repository; reduce only if memory requires, and document deviation |
| Budget | 20 epochs | Direct point of comparison in Table 8 |
| Random seed | 3407 | Report and repository; log all relevant software RNGs |
| Evaluation selection | Use group-disjoint **train-derived validation** for checkpoint choice | Adaptation to ensure final test isolation; NOT an exact reproduction of GitHub script |
| Final test | Published `eval_binary.csv` **only if verified as held-out** | Evaluate once after model/hyperparameters frozen |
| Run environment | Google Colab GPU or university GPU, versions logged | Different hardware from paper's A100; do NOT compare training wall time directly |

**Important:** The reference script monitors the `eval_label_path` each epoch and retains the checkpoint with highest evaluation accuracy. We must verify whether this is the final test set. Our clean evaluation uses a separate train-derived validation partition; report any difference as an **adapted protocol**. If no group IDs exist, do not pretend a random image-level validation split is independent. A fixed-epoch, no-test-tuning protocol can be a documented fallback, but then model selection limitations remain.

**Early stopping discrepancy:** Report describes a successive-loss-change threshold; GitHub code implements mean-training-loss ≤ `eps` (default `1e-5`). To compare to the 20-epoch reference, begin with fixed 20 epochs unless the dataset audit indicates otherwise; document that decision.

## C. Minimum reproducible pipeline

1. Inspect archive filenames, class labels, CSV schema, timestamps, duplicates, trial grouping, class counts and radar-frequency distribution.
2. Save an immutable manifest and checksums; record dataset version and source URL. Keep large data outside Git.
3. Audit original train/eval partitions, group overlap and exact image count. Only then create (if possible) group-disjoint validation using **train rows alone**.
4. Implement transforms exactly as configured, visualize images after transforms, and test tensor dimensions and labels.
5. Instantiate ResNet18 with two outputs and ImageNet initialization; verify the model supports a forward pass before full training.
6. Save config, versions, seeds, original source commit, code commit, learning curves, best checkpoint chosen by validation, and class counts.
7. Evaluate final holdout after model decisions are frozen. Log test accuracy, per-class precision/recall/F1, confusion matrix, UAV false-positive rate on non-UAV observations, and inference latency with hardware/batch details.
8. Compare the published 20-epoch ResNet18 result with ours in a table that explicitly identifies what matches and what does not. Never imply equal conditions if a validation procedure, image set, environment or code is different.

## D. Mandatory experiment log fields

`run_id`, `date`, `dataset_version`, `archive_checksum`, `original_code_commit`, `our_code_commit`, `split_file_hashes`, `trial_grouping_basis`, `task`, `label_mapping`, `image_transform`, `model`, `init_weights`, `optimizer`, `lr`, `batch_size`, `epochs_completed`, `seed`, `GPU`, `Python/PyTorch/torchvision versions`, `checkpoint_rule`, `validation_metrics`, `final_test_metrics`, `known_deviations`, `notes`.

Suggested run name: `SA_BIN_R18_PT20_3407_v1` (do not treat as an executed experiment yet).

## E. Sunday/Monday milestones

**Sunday 2026-11-15 — finish checkpoint 3:** dataset acquisition or partial extraction, provenance/CSV audit, frozen candidate protocol, repository skeleton, package environment, dataset loader, binary model, smoke test, baseline command with dry-run capability. Start a small training run only after validation/test isolation is resolved.

**Monday 2026-11-16 — checkpoint 4:** training, monitoring and final model evaluation; literature comparison and reproducible results log. If archive access/transfer blocks training, present the completed reproducibility audit and a smoke-tested pipeline rather than invented model metrics.

## F. First Codex prompt

> Read `docs/dataset_selection.md` and `docs/baseline_protocol.md`, then examine the reference GitHub project's `dataset.py`, `train.py`, `parse_args.py`, `constant.py`, and the St Andrews report. We are preparing **only** radar spectrogram classification (UAV/non-UAV), not a synthetic radar, trajectory tracker, or control algorithm. Do not invent files, classes, or radar data. First create a minimal PyTorch project skeleton and a **read-only dataset audit script** that examines the selected archive's CSV columns, image paths, class counts, source-trial identifiers, duplicate paths and split overlap. Do not train yet. Flag that the published `train.py` evaluates on `eval_binary.csv` every epoch and picks the best checkpoint using it; establish whether that is a test set. The reference transform is RGB → 224×224 resize → ToTensor(). Provide a short report of what was verified and what remains unknown. Follow existing project `docs/` conventions and add tests. Keep `data/` and model weights out of Git.

## G. Change control

If you need to alter a setting (e.g., resize, pretrained initialization, optimizer, batch size, splitting, augmentation), enter the change and its reason in a deviations table **before** treating the new result as comparable with the source report.

## H. Sources

- Uploaded St Andrews project report, pp. 8–14, tables 8–10.
- https://github.com/ZhongliangGuo/MathWorks-Radar-Drone-Project/blob/main/dataset.py
- https://github.com/ZhongliangGuo/MathWorks-Radar-Drone-Project/blob/main/train.py
- https://github.com/ZhongliangGuo/MathWorks-Radar-Drone-Project/blob/main/parse_args.py
