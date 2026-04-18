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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# PATHS
# =============================================================================
_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(_BASE_DIR, 'data', 'town04_dataset.csv')
MODEL_DIR    = os.path.join(_BASE_DIR, 'models')
RESULTS_DIR  = os.path.join(_BASE_DIR, 'results')
MODEL_PATH   = os.path.join(MODEL_DIR,   'lstm_sota_velocity.pth')
STATS_PATH   = os.path.join(MODEL_DIR,   'lstm_sota_stats.npz')
PLOT_PATH    = os.path.join(RESULTS_DIR, 'lstm_sota_training.png')
METRICS_PATH = os.path.join(RESULTS_DIR, 'lstm_sota_metrics.txt')

# =============================================================================
# CONFIGURATION
# =============================================================================
SEQ_LEN  = 40
STRIDE   = 2
FEATURE_COLS = ['ax_corr', 'ay_corr', 'wz', 'gt_speed_mps', 'gps_denied']
TARGET_COLS  = ['speed_next']

TRAIN_RUNS = [0, 1, 2]
VAL_RUNS   = [3]
HIDDEN1    = 64
HIDDEN2    = 32
DROPOUT    = 0.3

BATCH_SIZE    = 128
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 100
PATIENCE      = 15
GRAD_CLIP     = 1.0

# =============================================================================
# DATA PREPROCESSOR
# =============================================================================
class DataPreprocessor:
    def __init__(self):
        self.feat_mean = None; self.feat_std  = None
        self.tgt_mean  = None; self.tgt_std   = None

    def load_and_clean(self, path):
        df = pd.read_csv(path)
        df['speed_next'] = df['gt_speed_mps'].shift(-1)
        df.dropna(inplace=True)
        return df

    def fit(self, df_train):
        norm_feat_cols = [c for c in FEATURE_COLS if c != 'gps_denied']
        self.feat_mean = df_train[norm_feat_cols].mean().values.astype(np.float32)
        self.feat_std  = df_train[norm_feat_cols].std().values.astype(np.float32)
        self.feat_std  = np.where(self.feat_std < 1e-6, 1.0, self.feat_std)
        self.tgt_mean  = df_train[TARGET_COLS].mean().values.astype(np.float32)
        self.tgt_std   = df_train[TARGET_COLS].std().values.astype(np.float32)
        self.tgt_std   = np.where(self.tgt_std < 1e-6, 1.0, self.tgt_std)

    def transform_features(self, df):
        out = df.copy()
        norm_feat_cols = [c for c in FEATURE_COLS if c != 'gps_denied']
        out[norm_feat_cols] = (df[norm_feat_cols].values.astype(np.float32) - self.feat_mean) / self.feat_std
        return out

    def normalise_targets(self, arr):
        return (arr - self.tgt_mean) / self.tgt_std

    def denormalise_targets(self, arr):
        return arr * self.tgt_std + self.tgt_mean

    def save(self, path):
        np.savez(path, feat_mean=self.feat_mean, feat_std=self.feat_std, tgt_mean=self.tgt_mean, tgt_std=self.tgt_std,
                 feature_cols=np.array(FEATURE_COLS), target_cols=np.array(TARGET_COLS), seq_len=np.array([SEQ_LEN]))

# =============================================================================
# DATASET
# =============================================================================
class IMUSequenceDataset(Dataset):
    def __init__(self, df, prep, stride=1):
        self.X, self.y, self.tun = [], [], []
        feat_data = df[FEATURE_COLS].values.astype(np.float32)
        tgt_data  = prep.normalise_targets(df[TARGET_COLS].values.astype(np.float32))
        is_tun    = df['gps_denied'].values.astype(bool)

        for i in range(0, len(df) - SEQ_LEN, stride):
            self.X.append(feat_data[i : i+SEQ_LEN])
            self.y.append(tgt_data[i+SEQ_LEN - 1])
            self.tun.append(is_tun[i+SEQ_LEN - 1])

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx]), torch.tensor(self.tun[idx])

# =============================================================================
# MODEL (Uncertainty-Aware Velocity Predictor)
# =============================================================================
class UncertaintyVelocityPredictor(nn.Module):
    def __init__(self, input_size=5, h1=64, h2=32, dropout=0.3):
        super().__init__()
        self.input_ln = nn.LayerNorm(input_size)
        self.lstm1    = nn.LSTM(input_size, h1, num_layers=1, batch_first=True)
        self.ln1      = nn.LayerNorm(h1); self.drop1 = nn.Dropout(dropout)
        self.lstm2    = nn.LSTM(h1, h2, num_layers=1, batch_first=True)
        self.ln2      = nn.LayerNorm(h2); self.drop2 = nn.Dropout(dropout)
        # Outputs 2 values: [normalized_speed_pred, log_variance]
        self.head     = nn.Sequential(nn.Linear(h2, 16), nn.GELU(), nn.Linear(16, 2))

    def forward(self, x):
        x = self.input_ln(x)
        o1, _ = self.lstm1(x); o1 = self.ln1(o1); o1 = self.drop1(o1)
        o2, _ = self.lstm2(o1); o2 = self.ln2(o2); o2 = self.drop2(o2)
        out = self.head(o2[:, -1, :])
        mean_pred = out[:, 0].unsqueeze(1)
        log_var = out[:, 1].unsqueeze(1)
        return mean_pred, log_var

# =============================================================================
# LOSS: NLL (Negative Log Likelihood)
# =============================================================================
class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, mean_pred, log_var, target):
        # NLL Loss: 0.5 * exp(-log_var) * (obs - mu)^2 + 0.5 * log_var
        # Limits log_var to prevent explosion
        log_var = torch.clamp(log_var, min=-5.0, max=5.0)
        precision = torch.exp(-log_var)
        loss = 0.5 * precision * (target - mean_pred)**2 + 0.5 * log_var
        return loss.mean()

# =============================================================================
# TRAINING ENGINE
# =============================================================================
def run_epoch(model, loader, criterion, device, optimizer=None):
    if optimizer: model.train()
    else: model.eval()
    total_loss = 0.0
    
    for X, y, _ in loader:
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
    print("=== SOTA Velocity & Uncertainty Predictor ===")

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

    model = UncertaintyVelocityPredictor(input_size=len(FEATURE_COLS), h1=HIDDEN1, h2=HIDDEN2, dropout=DROPOUT).to(device)
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
            torch.save({'model_state': model.state_dict(), 'config': {'input_size': len(FEATURE_COLS)}}, MODEL_PATH)
        else: no_improve += 1
        
        print(f"Ep {epoch:3d} | Train NLL: {tr_loss:.4f} | Val NLL: {vl_loss:.4f} | Best Val: {best_val:.4f} | Patience: {no_improve}/15")
        if no_improve >= 15: break

    print(f"DONE. Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()
