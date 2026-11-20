# Dataset Selection — Radar-Based UAV / Non-UAV Classifier

**Decision date:** 2026-11-15 (UAE)  
**Decision ID:** DS-001  
**Status:** **Provisional selection, conditional on checking the distributed files**  
**Primary:** St Andrews / MathWorks Radar Drone Classification Project (2026 processed image dataset)  
**Fallback:** KTH 77 GHz radar dataset, version 2.0  
**Scope:** A classifier of observed radar signatures. Not radar search, range/angle estimation, target tracking, or a simulator controller.

## 1. Research question
Can a CNN classify an already observed radar signature as UAV vs non-UAV with a reproducible evaluation that explicitly distinguishes published-baseline reproduction from any adaptations?

## 2. Candidate comparison and weighted rubric

Scores are **researcher judgments**, not empirical performance. Scale 1=poor to 5=excellent. Total is sum(weight × score/5). The workbook `radar_dataset_selection_rubric.xlsx` contains the editable, formula-based version and evidence notes.

| Criterion | Weight | St Andrews | KTH | DIAT-μSAT | TSMS-Drone |
|---|---:|---:|---:|---:|---:|
| Alignment to UAV vs non-UAV task | 30% | 5 | 5 | 3 | 2 |
| Evaluation and split integrity | 25% | 3 | 2 | 1 | 2 |
| Literature / reproducibility | 20% | 3 | 5 | 5 | 4 |
| Implementation readiness | 15% | 5 | 2 | 5 | 1 |
| Actual dataset accessibility | 10% | 4 | 5 | 1 | 1 |
| **Weighted total (/100)** | **100%** | **80** | **76** | **60** | **43** |

**Why St Andrews is provisionally preferred:** Real UAV and bird recordings, clutter and noise; ready-made 512×512 micro-Doppler images; public code, reported baseline training settings, and a trial-independent split **claimed in its project report**. This fits the current research question and the limited implementation window. The direct report is a **technical report whose peer-review status is not verified**; its very high reported results are not evidence of generalization to new sensors, dates, or sites.

**Why KTH is the fallback:** Open data, real birds and drones, and a directly associated peer-reviewed IEEE RadarConf 2022 paper. However, the 1.6 GB dataset comprises complex-valued radar measurements requiring careful transformation into 5×150 network input, and the paper describes randomly splitting individual samples; recording-disjointness is not established.

**Why not DIAT-μSAT first:** Image files are easy to use and the literature is substantial, but the current IEEE DataPort access is restricted, non-UAV diversity is limited, the bird+rotor scene is not one object class, and raw acquisition/session information is absent. A 2026 audit showed sensitivity to image-level split protocols.

**Why not TSMS-Drone first:** Peer-reviewed dataset with processed images and multi-sensor potential, but the only non-UAV target is a reflector, acquisition distances are 2–30 m, and the available FMCW transfer was observed to be approximately 350 GB by the project researcher. The dataset is not well aligned with this short UAV-versus-bird experiment.

## 3. Report-derived evidence for St Andrews

- Three FMCW radars at 24, 94, 207 GHz; five drone models, six bird species. **Project report** pp. 4–6.
- Image construction produces 512×512 micro-Doppler images linked to timestamped raw recordings. **Project report** pp. 8–9.
- Before-image-generation trial-level split **as claimed**. **Project report** pp. 9–10.
- Selected **binary experimental** training set: 23,504 (11,752 UAV, 11,752 non-UAV); evaluation set: 2,146 (917 UAV, 1,229 non-UAV). **Project report** p. 10. These are NOT the full initial 51,032-image pool.
- Reference parameters: ResNet18 (among others), ImageNet initialization optional; learning rate 1e-5; batch size 64; random seed 3407; 20 and 100 epochs. **Project report** pp. 11–12; repository README.
- Authors report 100% binary accuracy for ImageNet-initialized ResNet18 after 20 epochs on their evaluation set. **Project report** Table 8, p. 11. This must be labeled the authors' **reported within-dataset evaluation** and not a guaranteed generalization result.

## 4. Code audit and experimental integrity warning (NEW)

The published GitHub implementation was checked on 2026-11-15:

1. `dataset.py` opens source images, converts to RGB, resizes to **224×224**, and calls `ToTensor()`; it maps binary labels from the column `drone_1_else_0`.
2. `train.py` evaluates `eval_label_path` **after each epoch**, and chooses the best saved weights by the accuracy on that same evaluation dataset. If `eval_binary.csv` is the final test set, this is **test-set-based checkpoint selection**, which biases the reported final test number. The actual role of that CSV must be established before labeling the result as independent testing.
3. The report says early stopping based on **change** in successive loss values, whereas repository `train.py` stops if average training loss is less than or equal to `args.eps` (whose default in `parse_args.py` is 1e-5). This mismatch must be recorded and not silently reconciled.
4. A trial-level split prevents some near-duplicate leakage, but birds and drones were recorded on different days. This may introduce environmental/acquisition confounding; it is a possible limitation, NOT proven bias.

## 5. Acceptance gates BEFORE claiming final selection

- [ ] 5.7 GB processed-image archive is readable and includes expected image files.
- [ ] Identify `train_binary.csv` and `eval_binary.csv`, class labels, actual row counts, image path mapping, duplicate files, and trial identifiers from timestamp folders or metadata.
- [ ] Verify that no trial/source file occurs across training and final test partitions; if metadata are insufficient, label this limitation explicitly.
- [ ] Confirm whether `eval_binary.csv` is the published test partition; reserve it untouched for **our** final evaluation.
- [ ] Create a separate **training-derived, group-disjoint validation partition**, if grouping metadata permit. Do not randomly split neighboring spectrograms.
- [ ] Verify that a sample image passes the reference preprocessing and produces a valid output through an untrained/pretrained ResNet18.

If these gates cannot be satisfied quickly, use KTH as the next dataset to inspect, rather than fabricating trial identifiers or reporting unverifiable comparability.

## 6. Decision and change log

| Date | Decision | Why / next verification |
|---|---|---|
| 2026-11-15 | St Andrews provisionally selected (80/100 vs KTH 76/100) | Task fit + ready images + research code; not yet a final accepted dataset |
| 2026-11-15 | Keep KTH as fallback | Direct peer-reviewed paper + open radar measurements |
| 2026-11-15 | Do not tune on published evaluation/test data | Preserve independent final test; code's eval-based checkpointing is a methodological caveat |
| 2026-11-15 | Initial model proposed: ImageNet-pretrained ResNet18, binary classification | Directly corresponds to published 20-epoch baseline; freeze after archive audit |

## 7. Key sources

- Guo, Rahman, Robertson, *Project Report: Machine Learning-based Drone and Bird Classification by Micro-Doppler Radar Signatures*, uploaded 23-page report, pp. 4–14.
- St Andrews GitHub: https://github.com/ZhongliangGuo/MathWorks-Radar-Drone-Project
- St Andrews Zenodo images: https://zenodo.org/records/18553708
- Karlsson et al. (2022), *Model-Aided Drone Classification Using Convolutional Neural Networks*, IEEE RadarConf: https://doi.org/10.1109/RadarConf2248738.2022.9764194
- KTH dataset: https://zenodo.org/records/5896641
- Karahan & Şanlıöz (2026), DIAT-μSAT archive audit: https://doi.org/10.3390/s26195989
- Han & Jung (2026), *Scientific Data*: https://doi.org/10.1038/s41597-026-06802-6
