# Experiment R-001: completed results and methodology

**Prepared:** 2026-11-16 (Asia/Dubai).  
**Current status:** Completed and evaluated.  
**Status review:** 2026-11-20 (Asia/Dubai); this is the documentation review date,
not an inferred training or evaluation date.

The original fixed-configuration decision follows `audit_review_checkpoint.md`
and the user's instruction; it supersedes the earlier protocol's proposed
validation split for R-001 only. The methodology and dated pre-training validation
records below are preserved. Their historical pending status does not describe
the completed experiment.

## Completed experiment

The authoritative project summary is [README.md](../README.md). The completed
run is recorded in [02_r001_colab_v3.ipynb](../notebooks/02_r001_colab_v3.ipynb),
whose saved outputs verify 20 logged epochs, checkpoint epoch 20,
`frozen_complete` status, the final checkpoint's presence and recorded hash,
and 2,146 saved evaluation predictions. The experiment was trained and evaluated
in Google Colab on an NVIDIA Tesla T4. This document update reviewed saved
outputs; it did not execute training, evaluation or notebook cells.

| Item | Completed R-001 |
|---|---|
| Model | ResNet18, ImageNet1K_V1 pretrained (`ResNet18_Weights.IMAGENET1K_V1`) |
| Dataset | MathWorks–St Andrews real radar spectrograms; original binary CSV partitions |
| Training | 23,504 images: 11,752 UAV and 11,752 non-UAV |
| Evaluation | 2,146 images: 917 UAV and 1,229 non-UAV |
| Epochs / batch size | 20 / 64 |
| Optimizer / learning rate | Adam / `1e-5` |
| Seed | 3407 |
| GPU | NVIDIA Tesla T4 |
| Recorded runtime | Python 3.13.16; PyTorch 2.11.0+cu130 |
| Final checkpoint | `final.pt` |
| Checkpoint selection | Predeclared final epoch (20), independent of evaluation accuracy |

The original training partition was used in full, with no new split. The
evaluation partition was reserved for the separate frozen-model evaluation;
`src/train.py` never opens it. The notebook records the first final evaluation
as a single post-freeze event and reuses complete saved results on subsequent
review. Do not restart or resume completed R-001, or use its evaluation results
to select hyperparameters or weights.

### Final evaluation results

These values are verified against the saved notebook evaluation summary and
README, rather than newly computed inference:

| Metric | Overall evaluation |
|---|---:|
| Accuracy | 100% |
| UAV precision | 100% |
| UAV recall | 100% |
| UAV F1-score | 100% |
| False positive rate on non-UAV | 0% |
| Incorrect predictions | 0 of 2,146 |

Confusion matrix: `[[1229, 0], [0, 917]]`. Rows are true labels and columns are
predictions, in order `[0 = non-UAV, 1 = UAV]`.

| Radar frequency | Evaluation images | Accuracy |
|---|---:|---:|
| 24 GHz | 519 | 100% |
| 94 GHz | 779 | 100% |
| 207 GHz | 848 | 100% |

These are results on the supplied evaluation split, not claims of perfect
training accuracy or independent-session generalization.

### Dataset-integrity findings

- Zero overlapping filenames between training and evaluation.
- Zero exact image-content duplicates across the partitions, as screened by
  SHA-256 hashes of encoded PNG bytes. This does not rule out near duplicates
  or identical pixels encoded differently.
- All non-UAV paths have source-folder date prefix `2024-02-06`; all UAV paths
  have `2024-03-01`, in both partitions. These dates are inferred from folder
  names, not independently verified acquisition timestamps.
- Possible recording-session similarity remains: the earlier audit found all
  15 UAV evaluation folders had a same-class, same-frequency training-folder
  timestamp within 30 seconds. This screening observation does not establish
  shared physical trials or overlapping observation intervals.

Class and folder date coincide, so acquisition effects cannot be separated from
class using this split. Neither date association nor timestamp proximity proves
leakage or that the model exploited acquisition artifacts. Conversely, exact
filename/byte disjointness does not establish independent recording sessions.

### Completed artifact locations

The notebook records the following paths under Google Drive root
`/content/drive/MyDrive/RadarClassifier/`:

| Relative Drive path | Artifact |
|---|---|
| `Runs/R-001/final.pt` | Frozen final-epoch model used for evaluation |
| `Runs/R-001/last.pt` | Recovery checkpoint; not an evaluation-based model choice |
| `Runs/R-001/frozen_config.json` | Frozen settings, environment and checkpoint identity |
| `Runs/R-001/training.jsonl` | Completed 20-epoch training history |
| `Evaluations/R-001-final/results.json` | Final metrics and provenance |
| `Evaluations/R-001-final/predictions.csv` | All 2,146 evaluation predictions |
| `Evaluations/R-001-final/r001_metrics_summary.csv` | Overall and per-frequency summary |
| `Evaluations/R-001-final/figures/r001_confusion_matrix.png` | Confusion matrix figure |

The saved evaluation summary records final-checkpoint SHA-256
`09a8522ed9a6e3d90c4ba0cb4866b094f61bedb49f674dd0cfe8e645851f917a`.
Its evaluation CSV SHA-256 matches `configs/r001.json`:
`f95ea124ab21410c4214d0ca62709207d97fba8eed36e9b5817a6a45440a3f83`.
The notebook's final verification output reports all listed artifacts present.
The Drive files themselves were not independently reopened during this review.
These are completed-run Drive locations; `outputs/` remains the local artifact
convention. Frozen weights and large datasets are not committed to GitHub or
bundled with the notebook.

## Frozen methodology

Use all 23,504 original training rows; retain all 2,146 evaluation rows exclusively
for a separate explicit final evaluation. CSV hashes in `configs/r001.json` are
copied from the completed integrity audit. No validation split or tuning occurs.

RGB conversion in Pillow, bilinear resize directly to 224x224 with antialiasing,
then float32 ToTensor scaling to [0,1]. No crop, augmentation or normalization.
This follows reference `dataset.py`, including its omission of ImageNet mean/std
normalization. It intentionally does not call the pretrained weights' recommended
inference transforms (which would add a resize/crop and normalization).

ResNet18 uses explicit `ResNet18_Weights.IMAGENET1K_V1`, equivalent to the inspected
torchvision ResNet18 DEFAULT. Weight URL ends in `resnet18-f37072fd.pth`.
Replace the ImageNet head with a randomly initialized biased Linear(512,2).
Fine-tune all layers, including normal BatchNorm training behavior. Record the
downloaded weight file's full SHA-256 before training. Downloads require the
explicit CLI flag if the file is not already cached.

20 epochs, batch size 64, seed 3407, Adam learning rate 1e-5, betas (0.9,0.999),
epsilon 1e-8, weight decay 0, AMSGrad false. Unweighted mean cross-entropy, no
label smoothing, scheduler, gradient clipping, mixed precision or early stopping.
Shuffle training each epoch using a seeded generator. Retain incomplete batches.
Use zero loader workers for Windows/Colab portability. Seed Python, NumPy and
PyTorch; request deterministic algorithms; disable cuDNN benchmarking and TF32.
Exact bitwise agreement across hardware/software is not guaranteed.

Predeclare **epoch 20** as the selected checkpoint. `last.pt` is a recovery
artifact containing model, optimizer, completed epoch, log history and random
states, atomically replaced each epoch. Explicit `--resume` is implemented as
described below. `final.pt` is written only upon completion.
Training never opens eval_binary.csv and never calls evaluation. Training logs
contain training loss (sample-weighted mean), accuracy, rows and epoch duration.
The run freezes config, source hashes, package versions, hardware, Git commit
(null if uninitialized), train CSV hash and final checkpoint hash in its output
directory (locally under `outputs/`; completed Drive locations are listed above).
Use a new output directory per run; existing runs require explicit `--resume`.

Evaluation validates completed status and checkpoint hash, uses argmax of two
logits (ties select class 0), and computes accuracy, UAV precision/recall/F1,
non-UAV false-positive rate and confusion matrix overall and by frequency.
Matrix rows are true labels, columns predicted, order [0=non-UAV,1=UAV].
Undefined ratios are JSON null. Predictions include p(UAV), filename and frequency.
Evaluation records the eval CSV hash, checkpoint identity and environment.
It was invoked separately after freeze; never use its output to tune R-001.

## Reference review and deviations

Inspected source commit: `78c899b1e36df3b0e8e19a390f34b9e209a5ccdb`.
Snapshots and file hashes are in `outputs/reference/`. Primary source:
[St Andrews repository](https://github.com/ZhongliangGuo/MathWorks-Radar-Drone-Project/tree/78c899b1e36df3b0e8e19a390f34b9e209a5ccdb).
Read local report pages 9-12: `../../Papers/MathWorks-Radar-Drone_Project - report.pdf`.
Report section 4.2 confirms LR, batch size, seed and the 20/100 epoch regimes;
Table 8 reports 100% binary accuracy for ImageNet-pretrained ResNet18 after 20
epochs. R-001 also achieved 100% on the supplied real-data evaluation split.
This matches the reference-reported accuracy while preserving the methodological
differences below. The additional R-001 precision, recall, F1 and false-positive
metrics must not be attributed to the reference without separate verification.
Optimizer details, transforms
and head are established from code, rather than inferred from the report.

| Item | Reference | R-001 and reason |
|---|---|---|
| Selection | Evaluate every epoch; save best evaluation accuracy | Final epoch only, preserving holdout isolation |
| Stopping | Report: successive loss change <1e-4; code: mean loss <= eps (default 1e-5) | Fixed 20 epochs, no early stopping; discrepancy preserved |
| Training batches | drop_last=True: 23,488 of 23,504 rows per epoch | drop_last=False: all 23,504 rows (368 batches, final batch 16) |
| Epoch loss log | Mean of batch means | Sample-weighted mean; no selection depends on it |
| Loader workers | Default 8; Windows comment recommends 0 | 0 for portability |
| RNG | torch.manual_seed | Additional seeds and deterministic settings, explicit loader generator; sequence may differ |
| Pretrained identity | weights='DEFAULT' | Explicit IMAGENET1K_V1, full weight hash logged |
| Software/hardware | Report A100 setup, original package versions unconfirmed | Historical local tests: CPU torch 2.6.0/torchvision 0.21.0; completed Colab run: Tesla T4, Python 3.13.16, torch 2.11.0+cu130 |

Adam defaults, head, transforms, full-network fine-tuning, loss and lack of scheduler
match inspected code. This is reference-aligned, not an exact reproduction.

## Remaining limitations

Trial independence and cross-sensor event correspondence remain unverified;
15 UAV eval folders have nearby training folders. Non-UAV and UAV folder-date
prefixes are respectively `2024-02-06` and `2024-03-01`, as detailed above.
These are path-derived date proxies. Date effects cannot be isolated from class.
Non-UAV subtype composition and historical authors' checkpoint used for Table 8
remain unresolved; current code verifies eval-based selection, not the historical
execution. No external-generalization claim follows from within-dataset metrics.
Archive checksum is unavailable; CSV and audited PNG hashes are available.
Pretrained ResNet18 training and frozen final evaluation are now completed on
the Tesla T4, with metrics recorded above. This does not establish portability
or bitwise reproducibility on other runtimes, cross-device recovery, or
Google Drive atomic-write durability. The saved outputs reviewed here do not
independently demonstrate every recovery path or reveal the full numerical
environment; retain the original run's environment records with the artifacts.

## Historical pre-training validation (2026-11-16)

The findings in this section describe the original implementation stage, before
the completed Colab experiment. They are retained as validation history, not
new tests run during this document update.

Ten unittests passed (five existing audit tests plus five classifier tests).
The synthetic smoke test uses two generated images, one ResNet18 optimizer step,
checkpoint round-trip and fixture inference/metrics with weights=None; a mock
rejects attempted weight downloads. It verifies that training needs no evaluation
CSV. Fixture source hashes remain unchanged. Additional checks cover exact
metric values, undefined denominators, transform scaling/dimensions, invalid
labels, deterministic RNG and refusal to evaluate incomplete training.
`pip check` and both CLI help commands passed.
Synthetic output: `outputs/r001_smoke_result.json` (not research results).

## Historical pre-training readiness cleanup (2026-11-16)

At this checkpoint, architecture, transforms, hyperparameters, original partitions
and final-epoch selection were unchanged, and real training/evaluation had not
yet been executed. The subsequent completed experiment is documented above.

`requirements-train.txt` is the canonical local training dependency file;
`requirements-training.txt` remains an alias. It pins torch 2.6.0 and torchvision
0.21.0, includes existing pandas 2.2.3/Pillow 11.1.0 audit pins, and explicitly
requires NumPy >=1.26.4,<3. The non-framework `requirements-runtime.txt` is suitable
for Colab; it leaves the preinstalled PyTorch/torchvision CUDA pair alone.
`python -m src.check_environment --colab --require-cuda` checks versions and an
untrained ResNet18 forward pass without downloading weights. Retained Colab
versions differing from the local baseline are logged software deviations;
successful preflight is an API check, not proof of identical numerical results.

`--resume <original-output-dir>/last.pt` requires the same output directory.
The completed checkpoint epoch is authoritative; continue from epoch+1, up to
the originally fixed epoch 20. No model initialization weights are downloaded
on resume. Legacy checkpoints lacking complete recovery state are refused.
Completed/frozen runs are refused. There is no mid-epoch resume: incomplete
epoch work is discarded and replayed from the last completed epoch.

Recovery includes model and Adam state, Python RNG, NumPy RNG, CPU torch RNG,
all CUDA RNG states and the DataLoader generator state. Restoration happens
after constructing/loading the model, optimizer and loader and before their
next iteration. Source hashes, Python/framework/numeric package versions,
CUDA/cuDNN/GPU identity, CUDA device count and selected device must match.
Relocating an unchanged project/dataset is allowed; changing runtime or code
requires a separate run rather than claiming exact continuation.

Before initial training and resume, verify CSV identity and compute an ordered
SHA-256 identity over every training filename and image byte hash. This can be
slow over Google Drive; it reads training data only. Images are assumed stable
during training. The evaluation CSV is never opened by either training path.

Checkpoints, logs and frozen JSON use temporary files in the same directory,
flush/fsync, then os.replace. A serialization failure preserves the old file.
The recovery checkpoint embeds completed log history, so resume repairs an
ahead/truncated text log. Atomicity/durability of mounted/cloud filesystems is
not guaranteed; never run two trainers against the same directory. Each resume
logs its environment and input checkpoint hash. Final checkpoint hashing and
explicit separate evaluation are preserved. Resume after a saved epoch 20 but
before final freeze simply regenerates the final artifact without more training.

Git detection tolerates a missing executable, timeout or uninitialized repo.
Source hashes and Git lookup resolve from the module's project root, regardless
of the process working directory. Default config resolution is also root-based.

Verified on local CPU: synthetic two-epoch interrupted/resumed ResNet18 matched
uninterrupted model/Adam/RNG tensors and training metrics exactly (timing excluded),
with no evaluation CSV present. Also tested identity/config/runtime mismatch
rejection, failed atomic serialization, dependency profiles and logging outside
the project with Git missing/uninitialized. At this pre-training stage, GPU RNG
restoration was implemented but not executed on the CPU-only machine; Colab
continuation, Google Drive atomicity, weight download and the full experiment
were then untested. The later Tesla T4 run establishes completed GPU training
and evaluation, without independently establishing all recovery/durability cases.
See README for current Colab review guidance and historical CLI commands, and
`outputs/r001_resume_smoke.json` for the synthetic comparison record.

Final readiness suite: **15 tests passed**; local standard and Colab-profile
CPU environment checks, pip check and offline requirements dry-run passed.
Additional image-byte mutation and legacy-checkpoint rejection tests passed.
Installed dependencies were not replaced during cleanup. Full validation log:
`outputs/r001_readiness_validation.md`.

## Conclusion

R-001 is completed and evaluated. The predeclared final-epoch ImageNet-pretrained
ResNet18 achieved 100% accuracy, UAV precision, recall and F1 with 0% false
positive rate on all 2,146 supplied evaluation images, matching the reference's
reported binary accuracy. The original configuration, partitions and documented
selection/stopping/batching decisions were preserved.

The result supports classification performance on this supplied split. Exact
deduplication does not resolve session independence, and the folder-date/class
association remains a confounding concern. Reliable generalization to independent
recordings, environments or sensors has not been demonstrated. Any follow-up
session-independent or external evaluation must be a separately declared study;
it must not retroactively tune or alter completed R-001.
