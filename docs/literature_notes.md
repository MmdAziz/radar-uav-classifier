# Literature Notes — Radar-Based UAV Classification

## Review 01: St Andrews / MathWorks Project Report

**Source:** Zhongliang Guo, Samiur Rahman, and Duncan A. Robertson, *Project Report: Machine Learning-based Drone and Bird Classification by Micro-Doppler Radar Signatures*, 23-page project report (including supplementary materials). University of St Andrews.  
**Online project:** https://github.com/ZhongliangGuo/MathWorks-Radar-Drone-Project  
**Review date:** 2026-11-14  
**Source type:** Project report and accompanying code repository; do not label this report peer-reviewed without independent confirmation.  
**Research task under consideration:** Binary classification of observed radar signatures into UAV / non-UAV.

### 1. Problem and method

- Investigates CNN classification of radar micro-Doppler imagery for (a) UAV/non-UAV, (b) drone type, and (c) drone/bird/clutter/noise tasks (pp. 1, 8).
- Real measurements were collected at the James Gregory Observatory using three FMCW radars: Blunderbuss (24 GHz), T-220 (94 GHz), and Theseus (207 GHz), plus synchronized 720p reference video (pp. 4–5).
- Targets include five drone models and six bird species, with distinct flight patterns and recording conditions. Drone trials included hovering, rotation, radial motion and some simultaneous drone flights (pp. 5–6).
- Radar measurements were annotated with video-based ground truth. Drone annotations used MATLAB Image Labeler; bird annotations involved manual video/radar alignment (pp. 7–8).

### 2. Dataset preparation

- For classification, the authors restricted the drone observations to **FMCW** data, aligning with the FMCW-only bird measurements (p. 8).
- They generated timestamp-associated **512 × 512** micro-Doppler images from annotated radar signals, described spectrogram window length of 256 samples, recorded generation settings, manually excluded unsuitable segments and cross-referenced labels (pp. 8–9).
- Preliminary generated image pool: **51,032** images: 12,669 drone; 38,363 non-drone. This is *not* the final experimental training/test count (p. 9).
- Three radar frequencies contributed 19,577 (94 GHz), 17,701 (207 GHz), and 13,754 (24 GHz) images to the preliminary pool (p. 9).

### 3. Data splitting and leakage prevention

- Authors explicitly report a **single-trial-independent** approach: assign original raw radar trials to partitions *before* generating their consecutive, potentially highly correlated spectrograms (pp. 9–10, Fig. 14).
- Final binary experimental set: training 23,504 (11,752 drone / 11,752 non-drone) and test 2,146 (917 drone / 1,229 non-drone), totaling **25,650** (p. 10).
- Final four-class experimental set: training 4,184 and test 455 (p. 10).
- **Outstanding checks:** Inspect actual CSVs and file paths to confirm group disjointness; clarify absence of an explicitly documented separate validation split; check repeated acquisition sessions, radar-frequency distribution and other confounders. Trial disjointness limits temporal leakage but is not by itself proof of generalization to new dates/sites/radars.

### 4. Models and setup

- Tested AlexNet, ResNet18/50, ResNeXt50, EfficientNetV2 and ConvNeXt variants (p. 11).
- Learning rate **1e-5**, batch size **64**, seed **3407**, and 20-/100-epoch regimes; initialization either random or ImageNet-pretrained (p. 12).
- The report states an early-stop criterion based on the difference between consecutive losses; the exact implementation and whether validation data inform stopping should be checked in code (p. 12).
- On their held-out real-data binary test, many architectures reached **100% reported accuracy** (Table 8, p. 11). This is a within-dataset result; it should not be represented as demonstrated real-world accuracy across new sites and sensors.

### 5. Real-to-simulated transfer and reverse transfer

- They generated **2,000 synthetic training** spectrogram images (1,000 per class) and **200 synthetic test** images (100 per class), using physical models of bird wing motion and drone rotor signatures (p. 11).
- Models **trained on real data and tested on synthetic data** show uneven class-specific performance. For example, ResNet18 with ImageNet pretraining and 20 epochs produced bird 97% and drone 88% under the binary-class entries in Table 9 (p. 12).
- Models **trained on synthetic data and tested on real data** can perform very differently: the ResNet18 pretrained/100-epoch example reported real bird **0%** and real drone **100%** in Table 10 (p. 13). These are *per-class* results, not overall balanced accuracy.
- All listed models achieved 100% on synthetic test data in the same synthetic distribution (Table 10, p. 13); this does **not** imply real-world transfer.
- The report attributes the gap to mismatch between simulated and real radar signatures, especially bird signatures (pp. 13–14).

### 6. Critical assessment and relevance to our project

**Strengths:** Relevant drone/bird/non-target classes; real radar measurements; processed images; several baselines; explicit trial-level split; simulations that can inform our later research.

**Cautions:** Very high within-dataset accuracy requires external-validation caution; trial partitions should be confirmed against distributed CSVs; bird and drone recording days differed (Feb 6 vs Mar 1, 2024), creating a *possible* acquisition-condition confound; test data are from the same overall collection campaign; training and validation procedures need code-level verification. Confounds are **hypotheses to check**, not demonstrated flaws.

**Scope:** The classifier recognizes cropped/processed radar spectrogram observations. It does **not** itself produce 3D object coordinates or demonstrate a live link to our colleague's coordinate-only synthetic radar.

### 7. Research decision — provisional

**Status: Shortlisted; not yet selected.** Compare against KTH. The report improves confidence in methodological intent, but we should inspect train/eval CSV provenance and grouping and confirm whether the processed image archive is sufficient for a reproducible baseline.

### 8. Follow-up checklist

- [ ] Review the GitHub training script, dataset loader and label CSV conventions.
- [ ] Inspect whether image filenames/timestamp folders permit trial-level split auditing.
- [ ] Verify included classes and sample counts in the actual downloadable processed image archive.
- [ ] Determine a non-test validation protocol for model tuning.
- [ ] Select a small reproducible model baseline (e.g., ResNet18) after dataset decision.
- [ ] Document the simulation–reality gap as a limitation in the final report.

---

## Review 02: KTH — Model-Aided Drone Classification Using CNNs (IEEE RadarConf 2022)

**Authors:** Alexander Karlsson, Magnus Jansson, Mikael Hämäläinen.  
**Paper DOI:** https://doi.org/10.1109/RadarConf2248738.2022.9764194  
**Dataset:** https://zenodo.org/records/5896641 (v2 `.npy`)  
**Review date:** 2026-11-15  
**Publication status:** Peer-reviewed IEEE conference paper.  
**Report location:** Uploaded 6-page PDF; figures 3–5 on pp. 5–6.

### Research question
How well does a CNN distinguish drones from non-drone radar targets when a drone type was **not represented during training**? Does a mathematically generated synthetic drone class improve the recognition of unfamiliar drones?

### Data and preprocessing
- 77 GHz SAAB SIRS 1600 FMCW radar; 75,868 scan segments in 130 recording groups; drone types D1–D6, bird, human, corner reflector.
- Uses 150 central FM sweeps and 5 range bins per example for a 5×150 input after spectrum conversion, Doppler centering and log normalization (paper pp. 2–3).
- Normalize SNR coverage from 3 to 30 dB via added complex Gaussian noise; augmentation/reuse stays within the original train/val/test assignment.
- Original paper selects a **random sample-level** split (1000 test, 1000 validation, remainder train per class before SNR augmentation); the public dataset includes split indicators. This is not proof of a recording-disjoint evaluation.

### Network and training
- Five-layer CNN: three convolutional and two fully connected layers; task-specific softmax (4, 5, 9, or 10 categories).
- Adam optimizer, learning rate 0.001, batch size 128, 25 epochs, He initialization; TensorFlow/Keras. Ten independently trained runs; model with best validation accuracy selected.
- Predicted drone subclasses are aggregated as the binary UAV decision; non-drone classes comprise bird, human and reflector.

### Main results and interpretation
- Nine-class drone-type / non-drone identification mean accuracy 0.68 (Figure 3). This is **not** UAV/non-UAV binary accuracy.
- When all drone types are represented in training, mean UAV class recall 0.88 and non-drone 0.93 (Figure 4).
- Unknown-drone mean binary recall increases from approximately 0.78 without synthetic drone class to 0.86 with the class: +8 percentage points (~10.3% relative) (Figure 4).
- At SNR >17.5 dB, mean classification across all trained classes exceeds 95%; unfamiliar-drone recall still lags and improves with the synthetic class (Figure 5).
- Paper emphasizes class generalization rather than end-to-end radar localization or tracking.

### Research limitations and implications
- The published partition is sample-level random; recording/session independence needs caution.
- Input transformation from raw complex radar samples is nontrivial and must be reproduced to compare directly.
- Accuracy varies with SNR and seen/unseen drone status; do not compare the nine-class identification number to St Andrews binary accuracy.
- Appropriate fallback if the St Andrews processed-image archive is unavailable or trial-split audit fails, but its different protocol prevents a direct numeric leaderboard.

### Decision
**Shortlisted as fallback.** Not chosen first due to implementation time and preprocessing burden. Directly peer-reviewed method is a major strength.

---

## Code Audit A1: St Andrews reference implementation (2026-11-15)

- `dataset.py` opens PIL images, converts RGB, resizes to 224×224, applies `ToTensor()`; for binary tasks the label is read from `drone_1_else_0`.
- `train.py` evaluates on `eval_label_path` every epoch and saves the checkpoint with the highest evaluation accuracy. If that evaluation file is the reported test set, the best-checkpoint rule introduces test-set selection bias. Verify its actual role before reproducing or interpreting scores.
- Project report describes stopping based on successive-loss changes; `train.py` stops when average training loss ≤ `eps`, default 1e-5 in `parse_args.py`. These differ and are recorded as a **source discrepancy**, not silently corrected.
- For our independent holdout, derive validation only from training recordings (if trustworthy grouping can be reconstructed) and do not inspect final test for model selection. Label this workflow a **methodological adaptation** rather than exact reproduction.
- Verified source: https://github.com/ZhongliangGuo/MathWorks-Radar-Drone-Project
