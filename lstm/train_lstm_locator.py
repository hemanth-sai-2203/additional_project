"""
train_lstm_locator.py - End-to-End Error-State LSTM Locator
============================================================
Architecture: EKF Pre-Processor -> LSTM Final Locator

The EKF smooths the raw IMU noise and provides a clean but drifting
trajectory. This LSTM learns the cumulative positional error pattern
(error_x, error_y) = (gt_position - ekf_position) and predicts it
so we can correct the EKF output in real-time.

Input features (per timestep):
    ekf_v      - EKF-smoothed velocity (m/s)
    ekf_psi    - EKF-smoothed heading (rad)
    ax_corr    - Raw corrected forward acceleration (m/s^2)
    ay_corr    - Raw corrected lateral acceleration (m/s^2)
    wz         - Raw gyroscope yaw rate (rad/s)
    pos_std_x  - EKF position uncertainty X (m)
    pos_std_y  - EKF position uncertainty Y (m)
    gps_denied - Binary flag (0 or 1)

Output targets:
    error_x = gt_x - ekf_x
    error_y = gt_y - ekf_y

At inference:
    final_x = ekf_x + lstm_predicted_error_x
    final_y = ekf_y + lstm_predicted_error_y
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# =============================================================================
# PATHS
# =============================================================================
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(_BASE_DIR, 'data', 'town04_physics_baseline_v2.csv')
MODEL_DIR   = os.path.join(_BASE_DIR, 'models')
RESULTS_DIR = os.path.join(_BASE_DIR, 'results')
MODEL_PATH  = os.path.join(MODEL_DIR,  'lstm_locator.pth')
STATS_PATH  = os.path.join(MODEL_DIR,  'lstm_locator_stats.npz')

# =============================================================================
# CONFIGURATION
# =============================================================================
SEQ_LEN  = 60   # 3 seconds of context (optimal for LSTM memory)
STRIDE   = 2    # Balanced redundancy

FEATURE_COLS = ['ekf_v', 'ekf_psi', 'ax_corr', 'ay_corr', 'wz',
                'pos_std_x', 'pos_std_y', 'gps_denied',
                'pitch_deg', 'roll_deg']
TARGET_COLS  = ['error_x', 'error_y']

TRAIN_RUNS = [0, 1, 2, 3]
VAL_RUNS   = [4]
TEST_RUNS  = [5]

# Robust model configuration — upgraded capacity
HIDDEN1  = 128
HIDDEN2  = 64
DROPOUT  = 0.3

BATCH_SIZE    = 256
LEARNING_RATE = 7e-4
WEIGHT_DECAY  = 1e-5
NUM_EPOCHS    = 200
PATIENCE      = 20
GRAD_CLIP     = 1.0


# =============================================================================
# DATA PREPROCESSOR
# =============================================================================
class DataPreprocessor:
    def __init__(self):
        self.feat_mean = None; self.feat_std = None
        self.tgt_mean  = None; self.tgt_std  = None

    def fit(self, df_train):
        # Normalise everything except binary gps_denied
        norm_cols = [c for c in FEATURE_COLS if c != 'gps_denied']
        self.feat_mean = df_train[norm_cols].mean().values.astype(np.float32)
        self.feat_std  = df_train[norm_cols].std().values.astype(np.float32)
        self.feat_std  = np.where(self.feat_std < 1e-6, 1.0, self.feat_std)
        self.tgt_mean  = df_train[TARGET_COLS].mean().values.astype(np.float32)
        self.tgt_std   = df_train[TARGET_COLS].std().values.astype(np.float32)
        self.tgt_std   = np.where(self.tgt_std < 1e-6, 1.0, self.tgt_std)

    def transform(self, df):
        out = df.copy()
        norm_cols = [c for c in FEATURE_COLS if c != 'gps_denied']
        out[norm_cols] = (df[norm_cols].values.astype(np.float32) - self.feat_mean) / self.feat_std
        return out

    def transform_array(self, arr):
        # Assumes input is (seq_len, num_features) or (num_features,)
        # FEATURE_COLS order: ['ekf_v', 'ekf_psi', 'ax_corr', 'ay_corr', 'wz', 'pos_std_x', 'pos_std_y', 'gps_denied', 'pitch_deg', 'roll_deg']
        # 'gps_denied' is index 7.
        out = arr.copy().astype(np.float32)
        if len(out.shape) == 1:
            # Single vector
            val_cols = np.delete(out, 7)
            norm_vals = (val_cols - self.feat_mean) / self.feat_std
            out[0:7] = norm_vals[0:7]
            out[8:] = norm_vals[7:]
        else:
            # Sequence (seq_len, 10)
            val_cols = np.delete(out, 7, axis=1)
            norm_vals = (val_cols - self.feat_mean) / self.feat_std
            out[:, 0:7] = norm_vals[:, 0:7]
            out[:, 8:] = norm_vals[:, 7:]
        return out

    def normalise_targets(self, arr):
        return (arr - self.tgt_mean) / self.tgt_std

    def denormalise_targets(self, arr):
        return arr * self.tgt_std + self.tgt_mean

    def save(self, path):
        np.savez(path,
                 feat_mean=self.feat_mean, feat_std=self.feat_std,
                 tgt_mean=self.tgt_mean, tgt_std=self.tgt_std,
                 feature_cols=np.array(FEATURE_COLS),
                 target_cols=np.array(TARGET_COLS),
                 seq_len=np.array([SEQ_LEN]))


# =============================================================================
# DATASET
# =============================================================================
class LocatorDataset(Dataset):
    """
    Creates sequences from CONTIGUOUS run segments.
    Each sample is a (SEQ_LEN, n_features) window mapped to the
    DELTA error (change in error from t-1 to t) at the last timestep.

    Key Design Decisions:
    - Delta target is stationary → generalizes across runs
    - FIX 1: Skip if prev/current are in different runs (boundary guard)
    - FIX 2: Compute delta in raw space FIRST, then normalize
      (norm(a) - norm(b) ≠ norm(a-b) due to mean subtraction)
    """
    def __init__(self, df, prep, stride=1, training=False):
        self.X, self.y, self.w = [], [], []
        self.training = training
        feat_data  = df[FEATURE_COLS].values.astype(np.float32)
        raw_tgt    = df[TARGET_COLS].values.astype(np.float32)  # raw space for delta
        run_ids    = df['run_id'].values
        is_tunnel  = df['gps_denied'].values.astype(np.float32)

        for i in range(0, len(df) - SEQ_LEN, stride):
            curr_idx = i + SEQ_LEN - 1
            prev_idx = i + SEQ_LEN - 2

            # FIX 1: Run boundary guard — skip if prev and current are in different runs
            if run_ids[curr_idx] != run_ids[prev_idx]:
                continue

            # FIX 2: Compute delta in raw space, then normalize
            current = raw_tgt[curr_idx]
            prev    = raw_tgt[prev_idx]
            delta   = current - prev
            delta_norm = prep.normalise_targets(delta)

            self.X.append(feat_data[i : i + SEQ_LEN])
            self.y.append(delta_norm)
            # Optimal tunnel weight for balanced RMSE
            self.w.append(3.0 if is_tunnel[curr_idx] > 0.5 else 1.0)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx].copy()
        
        # Robustness: Add noise jitter to IMU features only during training 
        if self.training:
            # Indices for ax_corr, ay_corr, wz are 2, 3, 4
            noise = np.random.normal(0, 0.05, (SEQ_LEN, 3)).astype(np.float32)
            X[:, 2:5] += noise

        return (torch.tensor(X),
                torch.tensor(self.y[idx]),
                torch.tensor(self.w[idx]))


# =============================================================================
# MODEL
# =============================================================================
class ErrorStateLSTM(nn.Module):
    """
    Deep LSTM that predicts (error_x, error_y) from a window of
    EKF-smoothed trajectory features.
    
    Architecture:
        Input(8) -> LayerNorm -> LSTM(128) -> LN -> Drop
                 -> LSTM(64) -> LN -> Drop -> last_step
                 -> Linear(64->32) -> GELU -> Linear(32->2)
    """
    def __init__(self, input_size=8, h1=128, h2=64, dropout=0.3):
        super().__init__()
        self.input_ln = nn.LayerNorm(input_size)
        self.lstm1    = nn.LSTM(input_size, h1, num_layers=1, batch_first=True)
        self.ln1      = nn.LayerNorm(h1)
        self.drop1    = nn.Dropout(dropout)
        self.lstm2    = nn.LSTM(h1, h2, num_layers=1, batch_first=True)
        self.ln2      = nn.LayerNorm(h2)
        self.drop2    = nn.Dropout(dropout)
        self.head     = nn.Sequential(
            nn.Linear(h2, 32), nn.GELU(), nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.input_ln(x)
        o1, _ = self.lstm1(x);  o1 = self.ln1(o1);  o1 = self.drop1(o1)
        o2, _ = self.lstm2(o1); o2 = self.ln2(o2);  o2 = self.drop2(o2)
        return self.head(o2[:, -1, :])


# =============================================================================
# TRAINING ENGINE
# =============================================================================
def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0; total_weight = 0.0

    for X, y, w in loader:
        X, y, w = X.to(device), y.to(device), w.to(device)
        if training: optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            pred = model(X)
            # Per-sample MSE
            per_sample = ((pred - y) ** 2).mean(dim=1)

            # Magnitude-weighted loss: forces model to care about large corrections
            # v2: Increased multiplier to 4.0 to aggressively target Max Error
            error_mag   = torch.norm(y, dim=1).detach()
            mag_weights = 1.0 + 4.0 * torch.clamp(error_mag, max=2.0)
            combined_w  = w * mag_weights

            loss = (per_sample * combined_w).sum() / combined_w.sum()

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss   += loss.item() * combined_w.sum().item()
        total_weight += combined_w.sum().item()

    return total_loss / total_weight


def evaluate_locator(model, df, prep, device):
    """
    Evaluate overall and tunnel-specific RMSE in real-world metres.
    Predictions are DELTA errors — must be accumulated per run,
    same logic as the live demo correction buffer.
    """
    model.eval()
    feat_data    = df[FEATURE_COLS].values.astype(np.float32)
    run_ids_arr  = df['run_id'].values

    preds_all = []
    with torch.no_grad():
        for i in range(SEQ_LEN, len(df)):
            seq = torch.tensor(feat_data[i - SEQ_LEN : i]).unsqueeze(0).to(device)
            p   = model(seq).cpu().numpy()[0]
            preds_all.append(p)

    preds      = np.array(preds_all)
    preds_real = prep.denormalise_targets(preds)

    df_eval      = df.iloc[SEQ_LEN:].reset_index(drop=True)
    run_ids_eval = run_ids_arr[SEQ_LEN:]

    # FIX: accumulate delta corrections per run AND mirror live demo logic
    # The LSTM correction is only applied during GPS denial.
    corr_x, corr_y = 0.0, 0.0
    prev_run = run_ids_eval[0]
    prev_gps_denied = False
    final_x_list, final_y_list = [], []

    for i in range(len(df_eval)):
        curr_run = run_ids_eval[i]
        gps_denied = df_eval.iloc[i]['gps_denied'] > 0.5

        if curr_run != prev_run:          # reset correction at run boundary
            corr_x, corr_y = 0.0, 0.0
            prev_run = curr_run
            prev_gps_denied = False

        # Live Demo Logic: Reset on GPS reconnect
        just_got_gps = prev_gps_denied and not gps_denied
        if just_got_gps:
            corr_x, corr_y = 0.0, 0.0

        # Live Demo Logic: Only accumulate/apply when GPS is denied
        if gps_denied:
            corr_x += preds_real[i, 0]
            corr_y += preds_real[i, 1]
            corr_x  = float(np.clip(corr_x, -50.0, 50.0))   # safety clip
            corr_y  = float(np.clip(corr_y, -50.0, 50.0))

        final_x_list.append(df_eval.iloc[i]['ekf_x'] + corr_x)
        final_y_list.append(df_eval.iloc[i]['ekf_y'] + corr_y)

        prev_gps_denied = gps_denied

    final_x = np.array(final_x_list)
    final_y = np.array(final_y_list)

    err = np.sqrt((final_x - df_eval['gt_x'].values)**2 +
                  (final_y - df_eval['gt_y'].values)**2)

    tun_mask  = df_eval['gps_denied'].values > 0.5
    road_mask = ~tun_mask

    ekf_err = np.sqrt((df_eval['ekf_x'].values - df_eval['gt_x'].values)**2 +
                      (df_eval['ekf_y'].values - df_eval['gt_y'].values)**2)

    def rmse(e): return float(np.sqrt(np.mean(e**2))) if len(e) > 0 else float('nan')

    return {
        'overall_rmse':     rmse(err),
        'tunnel_rmse':      rmse(err[tun_mask]),
        'tunnel_max':       float(np.max(err[tun_mask])) if tun_mask.any() else float('nan'),
        'road_rmse':        rmse(err[road_mask]),
        'ekf_tunnel_rmse':  rmse(ekf_err[tun_mask]),
        'ekf_overall_rmse': rmse(ekf_err),
        'improvement_pct':  (1 - rmse(err[tun_mask]) / rmse(ekf_err[tun_mask])) * 100 if tun_mask.any() else 0,
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 62)
    print("  End-to-End Error-State LSTM Locator")
    print("  Architecture: EKF Pre-Filter -> LSTM Error Predictor")
    print("=" * 62)

    # --- Load Data ---
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=['ekf_x', 'ekf_y', 'gt_x', 'gt_y', 'error_x', 'error_y'], inplace=True)
    print(f"Loaded {len(df):,} rows | runs: {sorted(df['run_id'].unique())}")

    df_train = df[df['run_id'].isin(TRAIN_RUNS)].reset_index(drop=True)
    df_val   = df[df['run_id'].isin(VAL_RUNS)].reset_index(drop=True)

    # --- Fit & Transform ---
    prep = DataPreprocessor()
    prep.fit(df_train)
    prep.save(STATS_PATH)

    df_train_n = prep.transform(df_train)
    df_val_n   = prep.transform(df_val)

    train_ds = LocatorDataset(df_train_n, prep, stride=STRIDE, training=True)
    val_ds   = LocatorDataset(df_val_n, prep, stride=1, training=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_ds):,} seqs | Val: {len(val_ds):,} seqs")
    print(f"Features: {FEATURE_COLS}")
    print(f"Targets:  {TARGET_COLS}")

    # --- Model ---
    model = ErrorStateLSTM(
        input_size=len(FEATURE_COLS), h1=HIDDEN1, h2=HIDDEN2, dropout=DROPOUT
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    criterion = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {param_count:,}")

    # --- Training Loop ---
    best_val = float('inf'); no_improve = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        vl_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)
        lr_now = optimizer.param_groups[0]['lr']

        if vl_loss < best_val:
            best_val = vl_loss; no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'config': {'input_size': len(FEATURE_COLS), 'h1': HIDDEN1, 'h2': HIDDEN2},
                'epoch': epoch,
                'val_loss': vl_loss,
            }, MODEL_PATH)
        else:
            no_improve += 1

        print(f"Ep {epoch:3d} | Train: {tr_loss:.6f} | Val: {vl_loss:.6f} | "
              f"Best: {best_val:.6f} | P: {no_improve}/{PATIENCE} | LR: {lr_now:.1e}")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # --- Final Evaluation ---
    print("\n" + "=" * 62)
    print("  FINAL EVALUATION")
    print("=" * 62)
    
    # Reload best model
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    
    for name, run_ids in [("Train", TRAIN_RUNS), ("Val", VAL_RUNS), ("Test", TEST_RUNS)]:
        df_split = df[df['run_id'].isin(run_ids)].reset_index(drop=True)
        df_split_n = prep.transform(df_split)
        metrics = evaluate_locator(model, df_split_n, prep, device)
        
        print(f"\n  [{name}] Runs {run_ids}")
        print(f"    EKF-Only Tunnel RMSE:  {metrics['ekf_tunnel_rmse']:.2f} m")
        print(f"    LSTM Locator Tunnel RMSE: {metrics['tunnel_rmse']:.2f} m")
        print(f"    Tunnel Max Error:      {metrics['tunnel_max']:.2f} m")
        print(f"    Road RMSE:             {metrics['road_rmse']:.2f} m")
        print(f"    Improvement:           {metrics['improvement_pct']:+.1f}%")

    print(f"\nModel saved to {MODEL_PATH}")
    print("=" * 62)


if __name__ == '__main__':
    main()
