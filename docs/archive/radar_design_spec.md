# Radar Subsystem Design Spec

Audience: a coding agent implementing the radar part of a counter-UAV simulation project.
Project: multi-agent airspace monitoring and interception (Junior Project, Group 06).

Read this whole file before writing code. If something is unclear or missing, **ask the human instead of inventing values**. Where this spec gives numbers, they are placeholders unless marked otherwise.

---

## 1. Goal

Build two independent modules that sit in front of the simulation environment:

| Module | What it is | Is it ML? |
|---|---|---|
| **A. Radar input classifier** | Takes radar micro-Doppler data and decides whether the target is an intruding UAV or something else (noise, bird, clutter). | Yes (CNN or similar) |
| **B. Synthetic radar** | Reads live ground-truth positions from the simulation and returns approximate, noisy coordinates, as a real radar would. | No (pure math) |

The two modules are separate. The classifier decides *what* a target is. The synthetic radar decides *where* it appears to be. A later step attaches the classifier's output to each radar track.

## 2. Scope

**In scope**
- Classifier trained on a public radar dataset.
- Synthetic radar measurement model, with configurable noise and detection behaviour.
- A simple tracker (Section 6), only after A and B work.
- A stable data interface so the simulation team can plug in without changes.

**Out of scope for now. Do not build.**
- Weapon-target assignment (WTA). It is delayed.
- Friend-or-foe or authorization logic. Assume every detected UAV is an intruder.
- Camera or vision detection. It belongs to the Senior phase.
- Clutter, multipath and building-shadow modelling. Leave hooks (Section 5.4) but do not implement.
- Real radar hardware, real encryption, real networking.
- Scenario generation. The scenario is fixed or set by the operator for now. Treat intruder positions as input.

## 3. System context

```
Simulation env (ground truth: intruder x,y,z, velocity)
        |
        v   every radar update step
Synthetic radar (Module B)  --->  noisy measurements (range, az, el, or xyz)
        |
        v
Tracker (optional, Section 6)  --->  tracks (position, velocity estimate)
        ^
        |  class + confidence per track
Classifier (Module A)  <---  micro-Doppler input (see 4.3 for how this connects)
        |
        v
Simulation env: interceptors fly toward the track estimates
```

The interceptors only ever see the radar output. They never see ground truth.

## 4. Module A: radar input classifier

### 4.1 Task
Binary decision per target: **UAV** vs **non-UAV** (noise, birds, other non-threat returns). Return a confidence in [0, 1].

### 4.2 Dataset
- Primary: **DIAT-uSAT**, a micro-Doppler spectrogram dataset of small aerial targets. As I understand it, it has about 4,849 spectrogram images, 6 classes (RC plane, three-short-blade rotor, three-long-blade rotor, quadcopter, bionic bird, mini-helicopter plus bionic bird) and was collected with an X-band CW radar. License is CC BY 4.0.
- **Access is not automatic.** Files sit behind an IEEE DataPort subscription, with a free educational route by email request from an institutional address. The human is arranging access. If the dataset is not in the working directory when you start, stop and ask. Do not substitute random data.
- Cite: Kumawat et al., "DIAT-uSAT: Small Aerial Targets' Micro-Doppler Signatures and Their Classification Using CNN," IEEE GRSL, 2022.

### 4.3 Class mapping
The dataset has no pure "noise" class. Build the binary labels like this and **document the mapping in the code**:
- **UAV:** quadcopter, rotor classes, mini-helicopter (confirm the exact set with the human).
- **Non-UAV:** bionic bird, RC plane (confirm), plus synthetic noise-only and clutter-like spectrograms you generate (for example, Gaussian noise textures and spectrograms with the target component removed).
- Report the class balance. If it is badly skewed, handle it with class weights or resampling and say which.

Before training, **ask the human whether RC plane and the helicopter classes should count as UAV or non-UAV.** This changes the results.

### 4.4 Method
- Preprocess: resize spectrograms to a fixed shape, normalise, grayscale or single-channel.
- Start with a small CNN trained from scratch. Only move to a pretrained backbone if the baseline is clearly weak.
- Split train/validation/test by **recording session or target, not by random image**, if the dataset metadata allows. Random image splits leak near-duplicate frames and inflate accuracy. If metadata does not allow it, say so in the report.
- Fix random seeds. Save the trained weights and the exact preprocessing code.

### 4.5 Metrics to report
- Confusion matrix, precision, recall and F1 for the UAV class.
- False-alarm rate (non-UAV classified as UAV). This matters most operationally.
- Calibration check: do confidences of 0.9 mean about 90% correct? Report a reliability table.
- Do not report accuracy alone.

### 4.6 Interface (what the rest of the system calls)
```python
classify(spectrogram) -> {"label": "uav" | "non_uav", "p_uav": float}
```

### 4.7 Known limitation to state in the report
The classifier is trained on one radar's data in controlled open-field recordings. It will not necessarily transfer to other radars or environments. Do not claim otherwise.

## 5. Module B: synthetic radar

### 5.1 Purpose
Stand in for a real radar. Each update step it reads true target positions from the simulation and returns what a real radar would plausibly report: approximate, noisy, sometimes missing.

### 5.2 Inputs (read live each step)
```json
{
  "t": 12.4,
  "targets": [
    {"id": "T1", "pos": [x, y, z], "vel": [vx, vy, vz]}
  ]
}
```
Units: metres and seconds, right-handed frame, z up. The radar's own position is a config value, not hard-coded.

### 5.3 Measurement model
For each target, relative to the radar position:

1. Compute true spherical coordinates: `r = sqrt(dx²+dy²+dz²)`, `az = atan2(dy, dx)`, `el = asin(dz / r)`.
2. Check detectability. The target is **dropped** (no measurement) if any of these hold:
   - `r < r_min` or `r > r_max`
   - azimuth or elevation outside the field of view
   - a random draw fails the detection probability `p_d`
3. Add Gaussian noise: `r' = r + N(0, σ_r)`, `az' = az + N(0, σ_az)`, `el' = el + N(0, σ_el)`.
4. Convert back to Cartesian xyz and output **both** the spherical and the Cartesian values.
5. Output only at the radar's update interval, not every simulation tick.
6. Add **false alarms**: each update, draw a Poisson-distributed number (rate `λ_fa`) of spurious measurements at random positions inside the field of view. Mark none of them as false in the output. The downstream code must not be able to tell.
7. **Merge rule (required):** if two or more targets fall within one resolution cell (range difference below `res_r` **and** angular separation below `res_ang`), return **one** merged measurement at the centroid. Do not reveal the true count. The simulation later reveals the count when an interceptor gets close, but that logic lives in the environment, not here.

### 5.4 Configuration
All parameters in one config file (YAML or JSON), never hard-coded.

| Parameter | Placeholder default | Note |
|---|---|---|
| `radar_pos` | `[0, 0, 0]` | metres |
| `update_rate_hz` | 1 | confirm against datasheet |
| `r_min`, `r_max` | 50 m, 3000 m | placeholder |
| `fov_az`, `fov_el` | 360°, 0°–60° | placeholder |
| `sigma_r` | 5 m | placeholder |
| `sigma_az`, `sigma_el` | 1° | placeholder |
| `p_d` | 1.0 | keep as a parameter even though it defaults to perfect |
| `lambda_fa` | 0 | keep as a parameter even though it defaults to none |
| `res_r`, `res_ang` | 30 m, 3° | placeholder |
| `dropout_prob` | 0.0 | per-update track dropout, default off |
| `seed` | fixed | for reproducibility |

**Replace placeholders with values from a real radar datasheet** and cite it. Ask the human which radar to base this on if none is chosen. `p_d`, `lambda_fa` and `dropout_prob` default to ideal for the Junior phase, but they must exist as settings so Senior can turn them on without rewriting anything.

### 5.5 Output
```json
{
  "t": 12.4,
  "measurements": [
    {"meas_id": "M7",
     "range": 812.3, "azimuth": 0.41, "elevation": 0.12,
     "pos_est": [x, y, z]}
  ]
}
```
Do not include ground-truth IDs in the output. The downstream consumer must not learn which measurement belongs to which true target.

### 5.6 Tests
- Zero noise, `p_d = 1`, no false alarms: output equals ground truth within floating-point error.
- Statistical test: over many samples the measurement error has the configured mean and standard deviation.
- Targets outside range or field of view are never reported.
- Two targets inside one resolution cell produce one measurement; outside it, two.
- Same seed gives identical output.

## 6. Tracker (optional, only after A and B pass their tests)
Turn per-step measurements into tracks with position and velocity estimates.
- Start with a constant-velocity Kalman filter and nearest-neighbour data association with a gating distance.
- Track lifecycle: tentative, confirmed, dropped after N missed updates.
- Output per track: `track_id`, `pos_est`, `vel_est`, `covariance`, `p_uav` (from Module A), `age`, `missed_updates`.
- Do not use machine learning here.

## 7. Deliverables
1. `classifier/`: training script, saved weights, `classify()` function, evaluation report with the metrics in 4.5.
2. `synthetic_radar/`: the module, the config file, the tests in 5.6.
3. A short README per module: how to run, how to change parameters, known limitations.
4. A sample script that runs a fixed intruder trajectory through the radar and plots true vs measured positions.

## 8. Rules for the agent
- Ask before assuming. Missing dataset, unclear class mapping and unknown radar choice are all questions, not guesses.
- Do not report or imply results you did not run. State the evaluation setup next to every number.
- Keep it simple: no frameworks beyond what is needed (NumPy, PyTorch or similar, matplotlib).
- Keep the interfaces in Sections 4.6 and 5.5 stable. Other team members are coding against them.

## 9. Open questions for the human
- Which dataset classes count as UAV vs non-UAV?
- Which real radar's datasheet should the synthetic parameters follow?
- Should the radar update rate match the simulation tick or run slower?
- Is the intruder's trajectory scripted, operator-driven or random?
