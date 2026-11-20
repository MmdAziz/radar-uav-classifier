# Radar UAV / non-UAV classifier research

Experiment R-001 is complete: an ImageNet-pretrained ResNet18 classifies
MathWorks–St Andrews radar spectrograms as UAV (`1`) or non-UAV (`0`).
The completed-run record is [notebooks/02_r001_colab_v3.ipynb](notebooks/02_r001_colab_v3.ipynb).
Its saved outputs record training, frozen-checkpoint verification, evaluation,
and split-integrity checks. The results below describe the supplied evaluation
split; independent recording-session generalization has not been established.

## Experiment R-001 — Completed results

| Item | Completed experiment |
|---|---|
| Model | ResNet18; `ResNet18_Weights.IMAGENET1K_V1`; biased Linear(512, 2) head; all layers fine-tuned |
| Dataset | MathWorks–St Andrews real radar spectrograms; original binary CSV partitions |
| Training | 23,504 samples: 11,752 UAV and 11,752 non-UAV |
| Final evaluation | 2,146 samples: 917 UAV and 1,229 non-UAV |
| Settings | 20 epochs; batch size 64; Adam LR `1e-5`; seed `3407`; cross-entropy loss |
| Checkpoint selection | Predeclared final epoch (20), saved as `final.pt` |
| Hardware | Google Colab, NVIDIA Tesla T4 |
| Recorded runtime | Python 3.13.16; PyTorch 2.11.0+cu130 (not the local CPU dependency profile) |

| Evaluation subset | Samples | Accuracy | UAV precision | UAV recall | UAV F1 | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 2,146 | 100% | 100% | 100% | 100% | 0% |
| 24 GHz | 519 | 100% | 100% | 100% | 100% | 0% |
| 94 GHz | 779 | 100% | 100% | 100% | 100% | 0% |
| 207 GHz | 848 | 100% | 100% | 100% | 100% | 0% |

There were **zero errors on this supplied evaluation split**. The confusion
matrix is `[[1229, 0], [0, 917]]`, with true labels as rows and predictions as
columns, ordered `[non-UAV, UAV]`. These are evaluation results, not a claim of
100% training accuracy or real-world reliability. Values are recorded in the
notebook's saved evaluation summary; this README update did not rerun inference.

## Scientific interpretation and limitations

The integrity checks found no exact training/evaluation filename overlap and no
identical PNG byte hashes across partitions. These checks do not establish
independence of physical trials or detect all visually similar observations.

All non-UAV images have source-folder date prefix `2024-02-06`; all UAV images
have `2024-03-01`. These are date proxies from paths, not independently verified
recording timestamps. Class and source-folder date therefore coincide, creating
an acquisition-confounding concern. Possible closely related training/evaluation
recording sessions also remain a concern: the earlier audit found 15 evaluation
folders with same-class, same-frequency training neighbors within 30 seconds.
Timestamp proximity is a screening observation, not proof of overlapping trials
or that the network used acquisition artifacts.

R-001 demonstrates performance on the supplied split. Generalization to
independent recording sessions, unseen environments or other radar systems is
unverified. Original trial boundaries, clock synchronization and cross-sensor
correspondence are needed to resolve session independence. The binary CSVs do
not establish finer non-UAV subtypes. Any later validation/tuning study must use
training data alone and be declared as a separate experiment.

## Code, methodology and artifacts

| Location | Purpose |
|---|---|
| `notebooks/02_r001_colab_v3.ipynb` | Completed Colab record: restoration, checks, saved results, figures and reference comparison |
| `configs/r001.json` | Predeclared R-001 settings and original CSV SHA-256 identities |
| `src/audit_dataset.py` | Read-only CSV, image and partition-integrity audit |
| `src/experiment.py` | Dataset loading, transforms, ResNet18, training/metric helpers and environment logging |
| `src/train.py`, `src/recovery.py` | Training and explicit interrupted-run recovery; never open the evaluation CSV |
| `src/evaluate.py` | Separate explicit evaluation of a completed, frozen `final.pt` |
| `src/check_environment.py`, `tests/` | Dependency/API checks and unit/synthetic-fixture tests |
| `docs/experiment_r001.md` | Pre-training methodology, reference review, deviations and readiness history |

The dated statements in the experiment notes describing training as pending are
historical preparation records. The completed notebook and this results section
provide the current status; configuration and experiment history are unchanged.

Preprocessing is Pillow RGB conversion → bilinear 224×224 resize with
antialiasing → `ToTensor()` scaling to `[0,1]`. There is no crop, augmentation or
ImageNet mean/std normalization, matching the inspected reference loader.
Adam uses betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0` and AMSGrad off.
Training retains the incomplete batch and uses no scheduler or early stopping.
The original training CSV is used in full; no new split was created.

R-001 is **reference-aligned**, with documented differences: the inspected
reference code monitored evaluation accuracy each epoch and selected its best
checkpoint; R-001 selected epoch 20 in advance and evaluated only after freeze.
R-001 retains the final incomplete training batch (`drop_last=False`), whereas the reference implementation discards it (`drop_last=True`). R-001 also disables early stopping; the reference report and code describe different stopping criteria.
See [experiment notes](docs/experiment_r001.md) and the
[reference implementation](https://github.com/ZhongliangGuo/MathWorks-Radar-Drone-Project/tree/78c899b1e36df3b0e8e19a390f34b9e209a5ccdb).
Matching the reported split accuracy does not imply exact replication.

The notebook records the completed artifacts under Google Drive:

- `RadarClassifier/Runs/R-001/`: `final.pt`, recovery `last.pt`,
  `frozen_config.json` and `training.jsonl`.
- `RadarClassifier/Evaluations/R-001-final/`: `results.json`, `predictions.csv`,
  `r001_metrics_summary.csv` and `figures/r001_confusion_matrix.png`.

These are Drive-relative locations from the notebook, not repository folders.
Generated local artifacts belong under `outputs/`. Frozen checkpoints and large
raw datasets are **not committed to GitHub**; the notebook does not bundle them.
An independent reviewer needs authorized dataset access and the saved checkpoint.
Recorded configuration, environment, CSV/source hashes and checkpoint hashes
identify the run. The notebook's final-checkpoint SHA-256 is
`09a8522ed9a6e3d90c4ba0cb4866b094f61bedb49f674dd0cfe8e645851f917a`.

## Installation and tests

### Windows PowerShell

Run from the workspace root. Python 3.13 was tested locally. Use the virtual
environment executable directly; activation or execution-policy changes are
unnecessary. For the audit only:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_audit_dataset.py -v
```

`requirements.txt` preserves the pandas/Pillow audit pins. For the locally tested
CPU training/test environment (torch 2.6.0 / torchvision 0.21.0):

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe -m src.check_environment
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`requirements-training.txt` remains an alias of `requirements-train.txt`.
The training profile explicitly includes NumPy; the completed Colab run used a
different framework build. For Windows CUDA, use a compatible pair/driver from
the [official version instructions](https://pytorch.org/get-started/previous-versions/)
instead of CPU wheels. Do not assume bitwise agreement across runtimes.

### Google Colab: review the completed run

Open `notebooks/02_r001_colab_v3.ipynb` and review its saved outputs first.
The notebook documents Drive mounting, project restoration to
`/content/r001_workspace`, dataset restoration to `/content/radar_r001/drone_project`,
and the existing artifact paths above. Its archive password prompt does not
save the password; obtain it from the dataset's official repository instructions.

For an active notebook session, retain Colab's CUDA-compatible PyTorch/torchvision
pair. Install `requirements-runtime.txt`, not the local pinned framework profile.
The notebook also installs `jedi` to address a missing dependency found in its
prior Colab environment. Follow its environment checks and restart the runtime
if already-imported packages change. From the restored project root:

```bash
python -m src.check_environment --colab --require-cuda
python -m pip check
python -m unittest discover -s tests -v
```

These checks do not download pretrained weights or run the real experiment.
Notebook figures use its plotting environment; the minimal CLI requirements do
not specify a separate plotting stack. GPU training and evaluation are confirmed
by the saved Tesla T4 outputs; cloud-filesystem durability and portability to
other runtimes are not established by that run.

**Do not restart or resume completed R-001.** The notebook reuses complete saved
evaluation artifacts when present. If they are absent, its evaluation cell can
launch inference; inspect paths and saved artifacts before executing that cell.
Any intentionally repeated evaluation needs a new output directory and should
be recorded as a separate evaluation event, never used for tuning R-001.

## Read-only dataset audit

The configurable dataset root must contain `train_binary.csv`, `eval_binary.csv`
and `data/`. CSV filenames resolve under `<data-root>/data/`; relative CLI paths
resolve from the current working directory. The verified local extraction has
an additional archive directory:

```powershell
.\.venv\Scripts\python.exe -m src.audit_dataset --data-root "../../RadarData/MathWorks-project-dataset/drone_project" --output-dir "outputs/audits"
```

If your extraction uses the originally proposed layout, substitute
`../../RadarData/drone_project`. Do not copy the dataset into the repository.
If access outside the workspace is restricted, grant read-only access or run
the command locally. Source images/CSVs are never rewritten; outputs must be
outside the dataset.

Each audit creates a new UTC-named directory under `outputs/audits/` containing:

- `audit_report.md`: counts, distributions, integrity findings and caveats.
- `summary.json`: full problem/overlap/temporal lists, commands, dependencies
  and CSV/script hashes.
- `manifest.csv`: valid PNG references, source CSV rows, labels, status,
  dimensions/mode and encoded-file SHA-256.

Unsafe paths are reported and never opened; their rows appear in JSON rather
than the image manifest. Unique referenced PNG paths are checked once using
Pillow `verify()` and full `load()`; repeated references are counted separately.
SHA-256 detects exact encoded-byte duplicates, not near duplicates or identical
pixels encoded differently. Unreferenced files and simulation data are excluded.
Folder timestamps are recording proxies; the fixed 30-second proximity screen
is not a tuned leakage rule.

Exit code `0` means the audit completed, not that every integrity check passed.
Code `2` means missing/unreadable CSVs, schema errors or invalid arguments
prevented completion. Read the report. To record an installed environment:

```powershell
New-Item -ItemType Directory -Force outputs/audits | Out-Null
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 outputs/audits/environment-lock.txt
```

## Historical CLI reference — do not rerun completed R-001

<details>
<summary>Training, recovery and separate evaluation commands (reference only)</summary>

The following are the preserved local CLI examples, **not instructions to repeat
the completed run**. `outputs/R-001` and `outputs/R-001-final-evaluation` are local
examples; the notebook's actual completed artifacts use the Drive paths above.

```powershell
# Initial training: creates a new run and may download ImageNet weights.
.\.venv\Scripts\python.exe -m src.train --config configs/r001.json --data-root "../../RadarData/MathWorks-project-dataset/drone_project" --output-dir outputs/R-001 --device auto --allow-weight-download

# Recovery: only for an incomplete run in its original output directory.
.\.venv\Scripts\python.exe -m src.train --data-root "../../RadarData/MathWorks-project-dataset/drone_project" --output-dir outputs/R-001 --resume outputs/R-001/last.pt --device auto

# Separate evaluation: only after final checkpoint/configuration freeze.
.\.venv\Scripts\python.exe -m src.evaluate --data-root "../../RadarData/MathWorks-project-dataset/drone_project" --run-dir outputs/R-001 --output-dir outputs/R-001-final-evaluation --device cpu
```

Training never accesses `eval_binary.csv`. Recovery restores model, Adam,
completed epoch, Python/NumPy/PyTorch/CUDA RNGs and DataLoader state, verifying
configuration, training CSV/image identity, source hashes and numeric runtime.
It continues from the next epoch; partial epochs are replayed. Pretrained weights
are not needed for recovery. Legacy incomplete-state checkpoints and completed
runs are refused. Initial training and recovery hash all training images, which
can be slow on Drive. Same-directory atomic replacements are used where
supported; do not run concurrent trainers against one directory.

Evaluation requires `frozen_complete` status and verifies `final.pt` and CSV
identity before inference. It writes `results.json` and `predictions.csv` to a
new output directory, separately from training/recovery artifacts. Predictions
use argmax over two logits; undefined metric denominators are reported as null.
The evaluation set is not used to choose checkpoints, epochs or hyperparameters.

</details>
