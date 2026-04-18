import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(_BASE_DIR, 'data', 'town04_dataset.csv')
MODEL_DIR    = os.path.join(_BASE_DIR, 'models')
RESULTS_DIR  = os.path.join(_BASE_DIR, 'results')
MODEL_PATH   = os.path.join(MODEL_DIR,   'lstm_physics_predictor.pth')
STATS_PATH   = os.path.join(MODEL_DIR,   'lstm_physics_stats.npz')

# =============================================================================
# CONFIGURATION
# =============================================================================
SEQ_LEN  = 40
STRIDE   = 2
# Pure IMU features! Completely decoupled from EKF speed feedback loop.
FEATURE_COLS = ['ax_corr', 'ay_corr', 'wz']
# 3D Bias targets (Forward Accel, Lateral Accel, Yaw Rate)
TARGET_COLS  = ['bias_fwd', 'bias_lat', 'bias_wz']

TRAIN_RUNS = [0, 1, 2, 3]; VAL_RUNS = [4]
HIDDEN1 = 64; HIDDEN2 = 32; DROPOUT = 0.3
BATCH_SIZE = 128; LEARNING_RATE = 1e-3; WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 150; PATIENCE = 15; GRAD_CLIP = 1.0

# =============================================================================
# DATA PREPROCESSOR
# =============================================================================
class DataPreprocessor:
    def __init__(self):
        self.feat_mean = None; self.feat_std  = None
        self.tgt_mean  = None; self.tgt_std   = None

    def load_and_clean(self, path):
        df = pd.read_csv(path)
        
        # 1. Forward and Lateral Gravity-Leak / Slip Biases
        df['bias_fwd'] = (df['gt_accel_fwd_mps2'] - df['ax_corr']).clip(-10.0, 10.0)
        df['bias_lat'] = (df['gt_accel_lat_mps2'] - df['ay_corr']).clip(-10.0, 10.0)
        
        # 2. Heading Drift (Gyro Bias)
        # Compute true wz via derivative of ground truth heading.
        dt = 0.05
        d_yaw = df['gt_heading'].diff()
        d_yaw = (d_yaw + np.pi) % (2 * np.pi) - np.pi # unwrap
        true_wz_raw = d_yaw / dt
        
        # Apply gentle smoothing because symbolic derivative amplifies tiny GT noise
        true_wz_smooth = true_wz_raw.rolling(window=5, min_periods=1).mean()
        
        df['bias_wz'] = (true_wz_smooth - df['wz']).clip(-1.0, 1.0)
        
        # Drop the first row (NaN from diff) and any others
        df.dropna(subset=['bias_wz', 'bias_fwd', 'bias_lat'], inplace=True)
        return df

    def fit(self, df_train):
        self.feat_mean = df_train[FEATURE_COLS].mean().values.astype(np.float32)
        self.feat_std  = df_train[FEATURE_COLS].std().values.astype(np.float32)
        self.feat_std  = np.where(self.feat_std < 1e-6, 1.0, self.feat_std)
        self.tgt_mean  = df_train[TARGET_COLS].mean().values.astype(np.float32)
        self.tgt_std   = df_train[TARGET_COLS].std().values.astype(np.float32)
        self.tgt_std   = np.where(self.tgt_std < 1e-6, 1.0, self.tgt_std)

    def transform_features(self, df):
        out = df.copy()
        out[FEATURE_COLS] = (df[FEATURE_COLS].values.astype(np.float32) - self.feat_mean) / self.feat_std
        return out

    def normalise_targets(self, arr): return (arr - self.tgt_mean) / self.tgt_std
    def denormalise_targets(self, arr): return arr * self.tgt_std + self.tgt_mean

    def save(self, path):
        np.savez(path, feat_mean=self.feat_mean, feat_std=self.feat_std, 
                 tgt_mean=self.tgt_mean, tgt_std=self.tgt_std,
                 feature_cols=np.array(FEATURE_COLS), target_cols=np.array(TARGET_COLS), 
                 seq_len=np.array([SEQ_LEN]))

# =============================================================================
# DATASET
# =============================================================================
class IMUSequenceDataset(Dataset):
    def __init__(self, df, prep, stride=1):
        self.X, self.y = [], []
        feat_data = df[FEATURE_COLS].values.astype(np.float32)
        tgt_data  = prep.normalise_targets(df[TARGET_COLS].values.astype(np.float32))

        for i in range(0, len(df) - SEQ_LEN, stride):
            self.X.append(feat_data[i : i+SEQ_LEN])
            # Target is the bias at the final step of the window
            self.y.append(tgt_data[i+SEQ_LEN - 1])

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# =============================================================================
# MODEL (3D Physics Uncertainty Predictor)
# =============================================================================
class PhysicsBiasPredictor(nn.Module):
    def __init__(self, input_size=3, output_size=3, h1=128, h2=64, dropout=0.3):
        super().__init__()
        self.input_ln = nn.LayerNorm(input_size)
        self.lstm1    = nn.LSTM(input_size, h1, num_layers=1, batch_first=True)
        self.ln1      = nn.LayerNorm(h1); self.drop1 = nn.Dropout(dropout)
        self.lstm2    = nn.LSTM(h1, h2, num_layers=1, batch_first=True)
        self.ln2      = nn.LayerNorm(h2); self.drop2 = nn.Dropout(dropout)
        # Outputs mean(3) and log_var(3)
        self.head     = nn.Sequential(nn.Linear(h2, 32), nn.GELU(), nn.Linear(32, output_size * 2))

    def forward(self, x):
        x = self.input_ln(x)
        o1, _ = self.lstm1(x); o1 = self.ln1(o1); o1 = self.drop1(o1)
        o2, _ = self.lstm2(o1); o2 = self.ln2(o2); o2 = self.drop2(o2)
        out = self.head(o2[:, -1, :])
        mean_pred = out[:, :3]
        log_var   = out[:, 3:]
        return mean_pred, log_var

# =============================================================================
# LOSS: NLL (Negative Log Likelihood) over 3 dimensions
# =============================================================================
class GaussianNLLLoss(nn.Module):
    def forward(self, mean_pred, log_var, target):
        log_var = torch.clamp(log_var, min=-5.0, max=5.0)
        precision = torch.exp(-log_var)
        # Sum NLL across the 3 output dimensions
        loss = 0.5 * precision * (target - mean_pred)**2 + 0.5 * log_var
        return loss.mean()

# =============================================================================
# TRAINING ENGINE
# =============================================================================
def run_epoch(model, loader, criterion, device, optimizer=None):
    if optimizer: model.train()
    else: model.eval()
    total_loss = 0.0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if optimizer: optimizer.zero_grad()
        
        with torch.set_grad_enabled(optimizer is not None):
            mean_pred, log_var = model(X)
            loss = criterion(mean_pred, log_var, y)
            if optimizer:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=== True SOTA Physics-Aware Bias Predictor ===")

    prep = DataPreprocessor(); df = prep.load_and_clean(DATA_PATH)
    df_train = df[df['run_id'].isin(TRAIN_RUNS)].reset_index(drop=True)
    df_val = df[df['run_id'].isin(VAL_RUNS)].reset_index(drop=True)

    prep.fit(df_train); prep.save(STATS_PATH)
    df_train_n = prep.transform_features(df_train)
    df_val_n   = prep.transform_features(df_val)

    train_ds = IMUSequenceDataset(df_train_n, prep, stride=STRIDE)
    val_ds   = IMUSequenceDataset(df_val_n, prep, stride=1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PhysicsBiasPredictor(input_size=len(FEATURE_COLS), output_size=len(TARGET_COLS), 
                                 h1=HIDDEN1, h2=HIDDEN2, dropout=DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = GaussianNLLLoss()

    best_val = float('inf'); no_improve = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        vl_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)
        
        if vl_loss < best_val:
            best_val = vl_loss; no_improve = 0
            torch.save({'model_state': model.state_dict(), 'config': {'input_size': len(FEATURE_COLS), 'h1': HIDDEN1, 'h2': HIDDEN2}}, MODEL_PATH)
        else: no_improve += 1
        
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch:3d} | Train NLL: {tr_loss:.4f} | Val NLL: {vl_loss:.4f} | Best Val: {best_val:.4f} | Patience: {no_improve}/15 | LR: {lr_current:.6f}")
        if no_improve >= 15: break

    print(f"DONE. Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()
