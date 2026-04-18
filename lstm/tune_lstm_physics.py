import os
import random
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# REPRODUCIBILITY (Fixed Seeding)
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# PATHS
# =============================================================================
_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(_BASE_DIR, 'data', 'town04_dataset.csv')
MODEL_DIR    = os.path.join(_BASE_DIR, 'models')
TUNING_DIR   = os.path.join(MODEL_DIR, 'tuning')
RESULTS_DIR  = os.path.join(_BASE_DIR, 'results')

STATS_PATH   = os.path.join(MODEL_DIR, 'lstm_physics_stats.npz')
FINAL_BEST_PATH = os.path.join(MODEL_DIR, 'lstm_physics_predictor.pth')

os.makedirs(TUNING_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# SEARCH SPACE & TUNING CONFIG
# =============================================================================
LEARNING_RATES = [5e-4, 7e-4, 9e-4]
DROPOUTS       = [0.2, 0.3, 0.4]
WEIGHT_DECAYS  = [0, 1e-5, 5e-5]

# Faster convergence settings for tuning
NUM_EPOCHS         = 80
PATIENCE           = 7
SCHEDULER_PATIENCE = 3
GRAD_CLIP          = 1.0

# Fixed Model Arch
HIDDEN1 = 64
HIDDEN2 = 32
SEQ_LEN = 40
STRIDE  = 2
BATCH_SIZE = 256
NUM_WORKERS = 0

FEATURE_COLS = ['ax_corr', 'ay_corr', 'wz']
TARGET_COLS  = ['bias_fwd', 'bias_lat', 'bias_wz']

TRAIN_RUNS = [0, 1, 2, 3]
VAL_RUNS   = [4]

# =============================================================================
# DATA PIPELINE (Reused from train_lstm_physics.py)
# =============================================================================
class DataPreprocessor:
    def __init__(self):
        self.feat_mean = None; self.feat_std  = None
        self.tgt_mean  = None; self.tgt_std   = None

    def load_and_clean(self, path):
        df = pd.read_csv(path)
        df['bias_fwd'] = (df['gt_accel_fwd_mps2'] - df['ax_corr']).clip(-10.0, 10.0)
        df['bias_lat'] = (df['gt_accel_lat_mps2'] - df['ay_corr']).clip(-10.0, 10.0)
        dt = 0.05
        d_yaw = df['gt_heading'].diff()
        d_yaw = (d_yaw + np.pi) % (2 * np.pi) - np.pi
        true_wz_raw = d_yaw / dt
        true_wz_smooth = true_wz_raw.rolling(window=5, min_periods=1).mean()
        df['bias_wz'] = (true_wz_smooth - df['wz']).clip(-1.0, 1.0)
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
    def save(self, path):
        np.savez(path, feat_mean=self.feat_mean, feat_std=self.feat_std, 
                 tgt_mean=self.tgt_mean, tgt_std=self.tgt_std,
                 feature_cols=np.array(FEATURE_COLS), target_cols=np.array(TARGET_COLS), 
                 seq_len=np.array([SEQ_LEN]))

class IMUSequenceDataset(Dataset):
    def __init__(self, df, prep, stride=1):
        self.X, self.y = [], []
        feat_data = df[FEATURE_COLS].values.astype(np.float32)
        tgt_data  = prep.normalise_targets(df[TARGET_COLS].values.astype(np.float32))
        for i in range(0, len(df) - SEQ_LEN, stride):
            self.X.append(feat_data[i : i+SEQ_LEN])
            self.y.append(tgt_data[i+SEQ_LEN - 1])

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# =============================================================================
# MODEL & LOSS
# =============================================================================
class PhysicsBiasPredictor(nn.Module):
    def __init__(self, input_size=3, output_size=3, h1=64, h2=32, dropout=0.3):
        super().__init__()
        self.input_ln = nn.LayerNorm(input_size)
        self.lstm1    = nn.LSTM(input_size, h1, num_layers=1, batch_first=True)
        self.ln1      = nn.LayerNorm(h1); self.drop1 = nn.Dropout(dropout)
        self.lstm2    = nn.LSTM(h1, h2, num_layers=1, batch_first=True)
        self.ln2      = nn.LayerNorm(h2); self.drop2 = nn.Dropout(dropout)
        self.head     = nn.Sequential(nn.Linear(h2, 32), nn.GELU(), nn.Linear(32, output_size * 2))

    def forward(self, x):
        x = self.input_ln(x)
        o1, _ = self.lstm1(x); o1 = self.ln1(o1); o1 = self.drop1(o1)
        o2, _ = self.lstm2(o1); o2 = self.ln2(o2); o2 = self.drop2(o2)
        out = self.head(o2[:, -1, :])
        return out[:, :3], out[:, 3:]

class GaussianNLLLoss(nn.Module):
    def forward(self, mean_pred, log_var, target):
        log_var = torch.clamp(log_var, min=-5.0, max=5.0)
        precision = torch.exp(-log_var)
        loss = 0.5 * precision * (target - mean_pred)**2 + 0.5 * log_var
        return loss.mean()

# =============================================================================
# CORE TRAINING LOOP
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

def train_task(args):
    """Trains a single model config and returns its metrics. Designed for multiprocessing."""
    lr, dropout, weight_decay, df_train_n, df_val_n, prep, run_idx, total_runs = args
    
    device = torch.device('cpu') # Force CPU for parallel workers
    torch.set_num_threads(2)     # 2 math threads per worker (4 workers * 2 = 8 active cores)
    set_seed(42) # Ensure fair start
    
    # Create datasets locally inside the process
    train_ds = IMUSequenceDataset(df_train_n, prep, stride=STRIDE)
    val_ds   = IMUSequenceDataset(df_val_n, prep, stride=1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = PhysicsBiasPredictor(input_size=len(FEATURE_COLS), output_size=len(TARGET_COLS), 
                                 h1=HIDDEN1, h2=HIDDEN2, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=SCHEDULER_PATIENCE)
    criterion = GaussianNLLLoss()

    best_val_nll = float('inf')
    best_epoch = 0
    train_loss_at_best = float('inf')
    no_improve = 0
    
    val_loss_history = []
    
    model_name = f"physics_lr{lr}_do{dropout}_wd{weight_decay}.pth"
    model_path = os.path.join(TUNING_DIR, model_name)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        vl_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)
        
        val_loss_history.append(vl_loss)
        
        # Checkpointing happens ONLY at best epoch
        if vl_loss < best_val_nll:
            best_val_nll = vl_loss
            best_epoch = epoch
            train_loss_at_best = tr_loss
            no_improve = 0
            
            # Save the best weights safely
            torch.save({
                'model_state': model.state_dict(), 
                'config': {'input_size': len(FEATURE_COLS), 'h1': HIDDEN1, 'h2': HIDDEN2}
            }, model_path)
        else:
            no_improve += 1
            
            
        if no_improve >= PATIENCE:
            break
            
        # Parallel progress log (every 5 epochs)
        if epoch == 1 or epoch % 5 == 0 or no_improve == 0:
            print(f"    [Config {run_idx}/{total_runs}] Epoch {epoch:2d} | Val NLL: {vl_loss:7.4f} | Best: {best_val_nll:7.4f}", flush=True)

    # Calculate Stability Window Score
    idx = best_epoch - 1
    start_idx = max(0, idx - 2)
    end_idx = min(len(val_loss_history), idx + 3)
    window = val_loss_history[start_idx : end_idx]
    stability = np.std(window) if len(window) > 1 else 10.0
    
    gen_gap = best_val_nll - train_loss_at_best
    
    print(f"[{run_idx}/{total_runs}] DONE: LR {lr:.1e} | DO {dropout} | WD {weight_decay:.1e} -> Best Val: {best_val_nll:.4f} (Ep {best_epoch})", flush=True)
    
    return {
        'lr': lr, 'dropout': dropout, 'weight_decay': weight_decay,
        'best_val_nll': best_val_nll,
        'best_epoch': best_epoch,
        'stability': stability,
        'gen_gap': gen_gap,
        'model_path': model_path,
        'epochs_run': len(val_loss_history)
    }

# =============================================================================
# MAIN TUNING SCRIPT
# =============================================================================
def main():
    import concurrent.futures
    print("=" * 62)
    print("  Physics-Aware Bias Predictor - Parallel Targeted Search")
    print(f"  Search Space: {len(LEARNING_RATES) * len(DROPOUTS) * len(WEIGHT_DECAYS)} configs")
    print("=" * 62)

    # Global Seed
    set_seed(42)

    # 1. Prepare Data (once in main process)
    prep = DataPreprocessor(); df = prep.load_and_clean(DATA_PATH)
    df_train = df[df['run_id'].isin(TRAIN_RUNS)].reset_index(drop=True)
    df_val = df[df['run_id'].isin(VAL_RUNS)].reset_index(drop=True)

    prep.fit(df_train); prep.save(STATS_PATH)
    df_train_n = prep.transform_features(df_train)
    df_val_n   = prep.transform_features(df_val)

    # 2. Build Tasks
    tasks = []
    total_runs = len(LEARNING_RATES) * len(DROPOUTS) * len(WEIGHT_DECAYS)
    current_run = 0
    
    for lr in LEARNING_RATES:
        for do in DROPOUTS:
            for wd in WEIGHT_DECAYS:
                current_run += 1
                tasks.append((lr, do, wd, df_train_n, df_val_n, prep, current_run, total_runs))

    # 3. Run Search in Parallel (4 workers using 8 cores total)
    print("\n🚀 Starting 4 parallel training workers... (This will max out your CPU!)\n", flush=True)
    results = []
    
    # Use spawn to prevent CUDA/multiprocessing lockups on Windows
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:
        for metrics in executor.map(train_task, tasks):
            results.append(metrics)

    # 3. Rank and Filter Results
    df_res = pd.DataFrame(results)
    
    # Calculate final score: Best Val NLL + 0.1 * Stability
    df_res['score'] = df_res['best_val_nll'] + 0.1 * df_res['stability']
    
    # Filter out models that peaked before epoch 8 (too early)
    df_valid = df_res[df_res['best_epoch'] >= 8].copy()
    
    if df_valid.empty:
        print("\nWARNING: All models peaked before epoch 8. Falling back to ranking all models.")
        df_valid = df_res.copy()
        
    df_valid = df_valid.sort_values(by='score', ascending=True).reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print("  🏆 TOP 5 CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Rank':<5} | {'LR':<8} | {'DO':<5} | {'WD':<8} | {'Best Ep':<7} | {'Val NLL':<8} | {'Stability':<9} | {'Score':<8} | {'Gen Gap':<8}")
    print("-" * 80)
    
    for i in range(min(5, len(df_valid))):
        row = df_valid.iloc[i]
        print(f"#{i+1:<4} | {row['lr']:<8.1e} | {row['dropout']:<5.2f} | {row['weight_decay']:<8.1e} | "
              f"{row['best_epoch']:<7.0f} | {row['best_val_nll']:<8.4f} | {row['stability']:<9.4f} | "
              f"{row['score']:<8.4f} | {row['gen_gap']:<8.4f}")

    # 4. Deploy Best Model
    best_config = df_valid.iloc[0]
    best_path = best_config['model_path']
    shutil.copy2(best_path, FINAL_BEST_PATH)
    
    print("\n" + "=" * 80)
    print(f"🚀 Best model deployed to: {FINAL_BEST_PATH}")
    print("=" * 80)

if __name__ == '__main__':
    main()
