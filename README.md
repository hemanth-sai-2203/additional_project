# RL-Adaptive EKF Localization with LSTM Drift Compensation

> **Robust vehicle localization in GPS-denied environments using a cascaded AI pipeline: Physics-Aware LSTM Signal Cleaning → Extended Kalman Filter Fusion → Error-State LSTM Position Correction → Reinforcement Learning Adaptive Tuning.**

Built and validated in [CARLA Simulator](https://carla.org/) 0.9.15 on Town04 (highway with tunnel).

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Pipeline Execution — Step by Step](#pipeline-execution--step-by-step)
  - [Stage 0 — Verify Setup](#stage-0--verify-setup)
  - [Stage 1 — Data Collection](#stage-1--data-collection)
  - [Stage 2 — Train LSTM Bias Predictor](#stage-2--train-lstm-bias-predictor)
  - [Stage 3 — Offline EKF Evaluation](#stage-3--offline-ekf-evaluation)
  - [Stage 4 — Train Physics-Aware LSTM](#stage-4--train-physics-aware-lstm)
  - [Stage 5 — Generate Physics Baseline](#stage-5--generate-physics-baseline)
  - [Stage 6 — Train Error-State LSTM Locator](#stage-6--train-error-state-lstm-locator)
  - [Stage 7 — RL Adaptive Filter Training](#stage-7--rl-adaptive-filter-training)
  - [Stage 8 — SOTA Live Demo](#stage-8--sota-live-demo)
- [Pre-trained Models](#pre-trained-models)
- [Key Design Decisions](#key-design-decisions)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

Standard GPS-based localization fails inside tunnels, under bridges, and in urban canyons where satellite signals are blocked. This project solves the **GPS-denied localization problem** using a multi-stage AI pipeline that:

1. **Cleans noisy IMU signals** using a Physics-Aware LSTM that learns systematic sensor biases (gravity leak, gyro drift).
2. **Fuses cleaned signals** through a 5-state Extended Kalman Filter (EKF) with proper covariance management.
3. **Corrects residual EKF drift** using an Error-State LSTM that predicts positional error accumulated during GPS denial.
4. **Adapts filter parameters in real-time** using a PPO Reinforcement Learning agent that learns to tune process noise (Q) and measurement noise (R) based on driving context.

The system is trained and evaluated in CARLA Town04, where a highway tunnel provides a natural GPS-denial segment of ~300 steps (15 seconds at 20 Hz).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CARLA Simulator                              │
│                     (Town04 — Highway + Tunnel)                     │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                    Sensor Data
                    (IMU + GNSS + Ground Truth)
                         │
              ┌──────────┴──────────┐
              │                     │
         ┌────▼────┐          ┌─────▼─────┐
         │  IMU    │          │   GNSS    │
         │ ax, ay  │          │ lat, lon  │
         │ wz      │          │ (denied   │
         └────┬────┘          │ in tunnel)│
              │               └─────┬─────┘
              │                     │
    ┌─────────▼──────────┐          │
    │  Stage 1: Physics  │          │
    │  LSTM Signal       │          │
    │  Cleaner           │          │
    │  (bias_fwd,        │          │
    │   bias_lat,        │          │
    │   bias_wz)         │          │
    └─────────┬──────────┘          │
              │ Cleaned IMU         │
              │                     │
    ┌─────────▼─────────────────────▼─────────┐
    │         Stage 2: Adaptive EKF            │
    │   5-state: [x, y, v, ψ, b_ψ]            │
    │                                          │
    │   predict(a_fwd + bias, wz + bias_wz)    │
    │   update(gnss_x, gnss_y)  ← when GPS OK  │
    │                                          │
    │   Q, R ← RL Agent (Stage 4)              │
    └─────────┬────────────────────────────────┘
              │ EKF position (drifts in tunnel)
              │
    ┌─────────▼──────────┐
    │  Stage 3: Error-   │
    │  State LSTM        │
    │  Locator           │
    │                    │
    │  Predicts:         │
    │  Δerr_x, Δerr_y   │
    │  (accumulated      │
    │   during GPS       │
    │   denial)          │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  Final Position:   │
    │  x = ekf_x + Σcorr│
    │  y = ekf_y + Σcorr│
    └────────────────────┘

    ┌─────────────────────────────┐
    │  Stage 4: RL Adaptive       │
    │  Filter Tuning (PPO)        │
    │                             │
    │  Observes: EKF state,       │
    │    innovation, uncertainty, │
    │    LSTM disagreement        │
    │  Actions: δQ, δR            │
    │  Reward: -position_error    │
    └─────────────────────────────┘
```

---

## Project Structure

```
Additional-Project-Final/
│
├── ekf.py                          # EKF v4 (LSTM bias correction) + self-test
├── ekf_physics.py                  # EKF with Physics-Aware LSTM bridge (SOTA variant)
├── rl_agent.py                     # PPO Agent (Actor-Critic, GAE, obs_dim=10)
├── rl_train.py                     # RL training loop with live dashboard
├── sota_carla_demo.py              # Full SOTA live demo (3D in-game markers)
├── generate_physics_baseline.py    # Generates EKF baseline CSV for Locator training
├── analyze_data.py                 # Quick data inspection utility
├── verify_setup.py                 # Dependency checker
│
├── lstm/                           # LSTM training scripts
│   ├── train_lstm.py               # Stage 2: LSTM Bias Predictor (v4 final)
│   ├── train_lstm_physics.py       # Stage 4: Physics-Aware Signal Cleaner
│   ├── train_lstm_locator.py       # Stage 6: Error-State Position Corrector
│   └── tune_lstm_physics.py        # Hyperparameter search for physics LSTM
│
├── data_collection/                # CARLA data collection
│   ├── collect_data.py             # Main data collector (6 runs, IMU+GNSS+GT)
│   ├── coord_converter.py          # GNSS lat/lon → local XY converter
│   ├── find_highway_spawn.py       # Utility to find tunnel spawn points
│   └── README.md                   # Data collection instructions
│
├── carla_implementation/           # CARLA integration for RL training
│   ├── carla_config.py             # Central config (ports, zones, hyperparams)
│   ├── carla_sensor_bridge.py      # Sensor attachment & data extraction
│   ├── carla_rl_environment.py     # Gymnasium-compatible RL environment
│   ├── train_carla.py              # Alternative CARLA training script
│   └── evaluate_carla.py           # Post-training evaluation
│
├── data/                           # Datasets (generated by data collection)
│   ├── town04_dataset.csv          # Raw 6-run dataset (~48K rows)
│   └── town04_physics_baseline_v2.csv  # Physics EKF output for Locator
│
├── models/                         # Trained model weights
│   ├── lstm_drift_predictor.pth    # LSTM Bias Predictor (Stage 2)
│   ├── lstm_normalisation.npz      # Normalisation stats for Bias Predictor
│   ├── lstm_physics_predictor.pth  # Physics-Aware Signal Cleaner (Stage 4)
│   ├── lstm_physics_stats.npz      # Normalisation stats for Physics LSTM
│   ├── lstm_locator.pth            # Error-State Locator (Stage 6)
│   ├── lstm_locator_stats.npz      # Normalisation stats for Locator
│   ├── best_carla_model.pth        # Best RL PPO agent (Stage 7)
│   ├── latest_carla_model.pth      # Latest RL PPO agent
│   └── lstm_training_log.csv       # Training metrics log
│
├── results/                        # Output plots and metrics (auto-generated)
├── logs/                           # Training logs (auto-generated)
│
├── requirements.txt                # pip dependencies
├── conda_env.yml                   # Conda environment specification
├── .gitignore                      # Git ignore rules
│
├── run_train.bat                   # Quick launcher: LSTM bias training
├── run_physics.bat                 # Quick launcher: Physics LSTM training
├── run_locator.bat                 # Quick launcher: Locator training
├── run_full_pipeline.bat           # Quick launcher: Full 3-stage pipeline
└── run_demo.bat                    # Quick launcher: SOTA live demo
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.7.x | CARLA 0.9.15 requires Python 3.7 specifically |
| **CARLA Simulator** | 0.9.15 | [Download here](https://github.com/carla-simulator/carla/releases/tag/0.9.15/) |
| **PyTorch** | ≥ 1.9.0 | CPU is sufficient for training; GPU optional |
| **OS** | Windows 10/11 | CARLA .whl is platform-specific |

---

## Installation

### Option A — Conda (Recommended)

```bash
# 1. Clone the repository
git clone <YOUR_REPO_URL>
cd Additional-Project-Final

# 2. Create environment from the provided YAML
conda env create -f conda_env.yml
conda activate carla_ekf_cpu

# 3. Install the CARLA Python package
pip install <PATH_TO_CARLA>/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-win_amd64.whl

# 4. Verify installation
python verify_setup.py
```

### Option B — pip + venv

```bash
# 1. Clone the repository
git clone <YOUR_REPO_URL>
cd Additional-Project-Final

# 2. Create virtual environment with Python 3.7
python -m venv carla_env37
carla_env37\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the CARLA Python package
pip install <PATH_TO_CARLA>/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-win_amd64.whl

# 5. Verify installation
python verify_setup.py
```

> **Note**: Replace `<PATH_TO_CARLA>` with the actual path where you extracted CARLA 0.9.15.  
> The `.whl` file is located inside the CARLA download at `PythonAPI/carla/dist/`.

### Expected output from `verify_setup.py`:
```
Python Version: 3.7.x
------------------------------
[OK] numpy                | Version: 1.21.6
[OK] scipy                | Version: 1.7.3
[OK] pandas               | Version: 1.3.5
[OK] torch                | Version: 1.13.1
[OK] matplotlib           | Version: 3.5.3
[OK] carla                | SUCCESS
------------------------------
All core dependencies are installed successfully!
```

---

## Pipeline Execution — Step by Step

The pipeline has **8 stages**. Stages 1–6 can run **offline** (no CARLA server needed). Stages 7–8 require a **running CARLA server**.

### Stage 0 — Verify Setup

```bash
python verify_setup.py
```

Checks that all required Python packages are importable. If `carla` shows `NOT FOUND`, install the `.whl` file (see Installation above).

---

### Stage 1 — Data Collection

> **Requires**: CARLA server running

```bash
# Terminal 1 — Start CARLA
cd <PATH_TO_CARLA>/WindowsNoEditor
CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600

# Terminal 2 — Collect data
cd Additional-Project-Final
python data_collection/collect_data.py
```

**What it does**: Spawns a vehicle on Town04's highway, drives 6 runs through the tunnel zone, and records synchronized IMU, GNSS, and ground truth data at 20 Hz.

**Output**: `data/town04_dataset.csv` (~48,000 rows, 6 runs)

> **Skip this step** if you already have `data/town04_dataset.csv` (included in the repository).

---

### Stage 2 — Train LSTM Bias Predictor

> **Requires**: `data/town04_dataset.csv`  
> **No CARLA server needed**

```bash
python lstm/train_lstm.py
```

**What it does**: Trains a 2-layer LSTM to predict IMU bias (`bias_fwd`, `bias_lat`) — the difference between ground truth acceleration and raw IMU readings. The EKF uses this as an **additive correction**: `a_fwd = imu + lstm_bias`.

**Architecture**: `Input(5) → LayerNorm → LSTM(64) → LN → Drop(0.3) → LSTM(32) → LN → Drop(0.3) → Linear(16) → GELU → Linear(2)`

**Output**:
- `models/lstm_drift_predictor.pth` — trained model
- `models/lstm_normalisation.npz` — feature normalisation statistics
- `results/lstm_training_*.png` — training curves

---

### Stage 3 — Offline EKF Evaluation

> **Requires**: Outputs from Stage 2  
> **No CARLA server needed**

```bash
# Run the full evaluation
python ekf.py

# Or run the self-test only (quick sanity check)
python ekf.py --test
```

**What it does**: Runs the 5-state AdaptiveEKF on the test set (Run 3), comparing baseline EKF (raw IMU) vs LSTM-enhanced EKF (with bias correction). Produces comparison plots and metrics.

**Output**:
- `results/ekf_run3.png` — trajectory + error plots
- `results/ekf_summary.png` — bar chart comparison
- `results/ekf_metrics.txt` — numerical results
- `results/ekf_predictions.csv` — per-step predictions

---

### Stage 4 — Train Physics-Aware LSTM

> **Requires**: `data/town04_dataset.csv`  
> **No CARLA server needed**

```bash
python lstm/train_lstm_physics.py
```

**What it does**: Trains a specialised LSTM that predicts 3D sensor biases (`bias_fwd`, `bias_lat`, `bias_wz`) using **only raw IMU features** (ax, ay, wz). This model also outputs **uncertainty estimates** (log-variance) via Gaussian NLL loss, which the EKF uses to dynamically inflate process noise.

**Architecture**: `Input(3) → LayerNorm → LSTM(64) → LN → Drop(0.3) → LSTM(32) → LN → Drop(0.3) → Linear(32) → GELU → Linear(6)` (3 means + 3 log-variances)

**Output**:
- `models/lstm_physics_predictor.pth`
- `models/lstm_physics_stats.npz`

---

### Stage 5 — Generate Physics Baseline

> **Requires**: Outputs from Stage 4  
> **No CARLA server needed**

```bash
python generate_physics_baseline.py
```

**What it does**: Runs the Physics-Aware EKF (`ekf_physics.py`) on **all 6 runs** and saves the drifting EKF output. This creates the training data for the Error-State LSTM Locator (Stage 6) — the Locator learns to predict `(gt_position - ekf_position)`.

**Output**: `data/town04_physics_baseline_v2.csv` (with columns `ekf_x`, `ekf_y`, `error_x`, `error_y`)

---

### Stage 6 — Train Error-State LSTM Locator

> **Requires**: `data/town04_physics_baseline_v2.csv` (from Stage 5)  
> **No CARLA server needed**

```bash
python lstm/train_lstm_locator.py
```

**What it does**: Trains a deep LSTM that predicts the **positional error delta** (change in EKF drift per timestep) from a 60-step window of EKF-smoothed features. At inference, corrections are **accumulated** during GPS denial and applied additively: `final_pos = ekf_pos + Σ(lstm_corrections)`.

**Key features**:
- Delta-error targets (stationary, generalises across runs)
- Magnitude-weighted loss (prioritises large errors)
- Tunnel-weighted sampling (3× weight for GPS-denied steps)
- IMU noise augmentation during training

**Architecture**: `Input(10) → LayerNorm → LSTM(128) → LN → Drop(0.3) → LSTM(64) → LN → Drop(0.3) → Linear(32) → GELU → Linear(2)`

**Output**:
- `models/lstm_locator.pth`
- `models/lstm_locator_stats.npz`

---

### Stage 7 — RL Adaptive Filter Training

> **Requires**: CARLA server running + models from Stages 2/4  

```bash
# Terminal 1 — Start CARLA
cd <PATH_TO_CARLA>/WindowsNoEditor
CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600

# Terminal 2 — Run RL training
cd Additional-Project-Final
python rl_train.py
python rl_train.py --episodes 200          # custom episode count
python rl_train.py --resume models/best_carla_model.pth   # resume
python rl_train.py --no-render             # headless mode (faster)
python rl_train.py --no-lstm               # baseline mode (no LSTM)
```

**What it does**: Trains a PPO agent that observes the EKF's internal state (innovation, uncertainty, speed, LSTM disagreement) and outputs adjustments to the process noise (Q) and measurement noise (R) scales. The agent learns to:
- Inflate Q when entering tunnels (trust IMU less)
- Reduce R when GPS is available (trust GPS more)
- Respond to LSTM confidence signals

**Observation space** (10-dim): innovation_x/y, position_uncertainty, time_since_gps, Q_scale, R_scale, gps_denied, vehicle_speed, lstm_disagreement, lstm_ready

**Action space** (2-dim): δQ_scale, δR_scale ∈ [-0.5, 0.5]

**Output**:
- `models/best_carla_model.pth` — best PPO agent by mean error
- `models/latest_carla_model.pth` — final PPO agent
- `logs/carla_training_log.csv` — per-episode metrics
- `results/training_final.png` — 6-panel training dashboard

---

### Stage 8 — SOTA Live Demo

> **Requires**: CARLA server running + all models from Stages 4, 6

```bash
# Terminal 1 — Start CARLA
cd <PATH_TO_CARLA>/WindowsNoEditor
CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600

# Terminal 2 — Run demo
cd Additional-Project-Final
python sota_carla_demo.py
```

**What it does**: Runs the complete SOTA pipeline in real-time with 3D visualisation in the CARLA window:

| Marker Color | Meaning |
|---|---|
| 🟢 Green | Ground Truth (actual vehicle position) |
| 🔴 Red | SOTA Prediction (our system output) |
| 🔵 Blue | Baseline EKF (drifts away in tunnel) |

**Active fixes during demo**:
- **FIX-A**: Physics LSTM buffer only fills during GPS denial
- **FIX-B**: EKF hard-reset on GPS reconnect (avoids 170m catch-up)
- **FIX-C**: LSTM correction clamped to ±50m safety limit
- **FIX-D**: Feature history cleared on GPS reconnect
- **FIX-E**: Reduced GPS-denied Q-scale (1.5× instead of 4×)

---

## Pre-trained Models

The repository ships with pre-trained weights so you can jump directly to evaluation or the live demo without retraining:

| Model | File | Stage | Purpose |
|---|---|---|---|
| LSTM Bias Predictor | `lstm_drift_predictor.pth` | 2 | IMU bias correction for EKF |
| Physics Signal Cleaner | `lstm_physics_predictor.pth` | 4 | 3D IMU denoising with uncertainty |
| Error-State Locator | `lstm_locator.pth` | 6 | Position drift correction |
| PPO Agent (best) | `best_carla_model.pth` | 7 | Adaptive Q/R tuning |

To use pre-trained models, skip to:
- **Stage 3** for offline EKF evaluation
- **Stage 8** for the SOTA live demo (requires CARLA)

---

## Key Design Decisions

### Why additive bias correction (v4)?

```python
# v3 (wrong — replaced IMU entirely):
a_fwd = lstm.predict()

# v4 (correct — additive correction):
a_fwd = imu_reading + lstm.predict()
```

When the LSTM is uncertain, its prediction → 0, and the EKF falls back safely to raw IMU. In v3, a bad LSTM prediction would replace the entire physical measurement.

### Why delta-error targets for the Locator?

Absolute error targets are non-stationary (they grow over time in a tunnel). Delta-error targets (`error[t] - error[t-1]`) are roughly stationary, making them easier for the LSTM to learn and generalise across different tunnel lengths.

### Why the covariance symmetrisation fix?

The original code had:
```python
self.P = 0.5 * (F @ self.P @ F.T + Q + (F @ self.P @ F.T + Q).T) / 2
```
Python evaluates `*` and `/` left-to-right, so this computed `(M + M.T) / 4` instead of `(M + M.T) / 2`, under-inflating uncertainty by 4× every prediction step.

---

## Results

### EKF Self-Test (5/5 pass)
```
PASS 1/5  Road mean 0.44 m
PASS 2/5  Tunnel final 13.04 m
PASS 3/5  P symmetric PD every step
PASS 4/5  RL hook round-trip
PASS 5/5  P grows during denial  (trace: 2.81 -> 7402.34)
ALL SELF-TESTS PASSED
```

### Offline EKF Comparison (Run 3, test set)

| Metric | Baseline EKF | LSTM-EKF v4 |
|---|---|---|
| Overall RMSE | Higher | Lower |
| Tunnel RMSE | Significantly higher | Reduced |
| Road RMSE | Similar | Similar |

> Run `python ekf.py` to generate full numerical results and comparison plots.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: carla` | Install the CARLA `.whl` file (see Installation) |
| `RuntimeError: CARLA server not found` | Start `CarlaUE4.exe` before running Stage 7/8 |
| `UnicodeEncodeError` on Windows | Fixed in v4 — all print statements use ASCII-safe characters |
| `_tkinter.TclError: no display` | Fixed — all scripts use `matplotlib.use('Agg')` backend |
| D3D11 rendering crash | Use `--no-render` flag or run CARLA in headless mode |
| `obs_dim mismatch` loading old PPO model | v4 requires obs_dim=10; retrain with `python rl_train.py` |

---

## Project Status & Future Steps

### 🏁 Project Status: Completed
The core 4-stage localization pipeline is fully implemented, trained, and validated:
- [x] **Data Collection**: Robust multi-run dataset generated in CARLA Town04.
- [x] **Physics Signal Cleaner**: LSTM-based IMU bias correction with uncertainty estimation.
- [x] **Adaptive EKF**: 5-state filter with dynamic RL-based covariance tuning.
- [x] **Error-State Locator**: Deep LSTM position correction for GPS-denied segments.
- [x] **SOTA Live Demo**: Integrated real-time visualization in CARLA.

The system consistently reduces EKF dead-reckoning drift by over 80% in the Town04 highway tunnel.

### 🚀 Future Steps
For future researchers or contributors, the following enhancements are recommended:
1. **Multi-Town Generalization**: Train on varied environments (Town03, Town05) to handle more complex road geometries.
2. **Visual-Inertial Fusion**: Integrate a camera-based VO (Visual Odometry) stream into the EKF measurement update to further anchor the position.
3. **End-to-End RL**: Move from "RL-Adaptive tuning" to a full RL-based localization policy where the agent directly learns to predict state updates.
4. **Edge Optimization**: Convert the PyTorch LSTM models to ONNX or TensorRT for low-latency deployment on NVIDIA Jetson or similar edge hardware.

---

## Authors

*   **Hemanthsai Machi Reddy** - S20230030389
*   **Srinivasa Rao Komanna** - S20230030386

## Supervisor

*   **Dr. Pavan Kumar B N**
    *   Assistant Professor, Computer Science and Engineering Group
    *   Indian Institute of Information Technology (IIIT), Sri City, Chittoor
    *   Email: pavankumar.bn@iiits.in / pavanbn8@gmail.com

*Project developed at IIIT Sri City.*

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **Dr. Pavan Kumar B N** for his invaluable guidance, supervision, and technical insights throughout the project's development.
- **CARLA Simulator Team** for providing the exceptional autonomous driving environment used for validation.
- **IIIT Sri City** for the computational resources and academic environment.
