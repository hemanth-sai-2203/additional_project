"""
generate_physics_baseline.py
=============================
Runs the FULL Physics-Aware EKF (with LSTM bias predictor for wz, ax, ay)
on all runs, then saves the drifting output so the LSTM Error State Corrector
can learn to fix the remaining drift.

Pipeline: CARLA data → Physics LSTM → EKF fusion → drifting baseline CSV
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ekf_physics import AdaptiveEKF, LSTMBridge, run_ekf_on_run

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(_BASE_DIR, 'data', 'town04_dataset.csv')
MODEL_PATH  = os.path.join(_BASE_DIR, 'models', 'lstm_physics_predictor.pth')
STATS_PATH  = os.path.join(_BASE_DIR, 'models', 'lstm_physics_stats.npz')
OUT_PATH    = os.path.join(_BASE_DIR, 'data', 'town04_physics_baseline_v2.csv')

def main():
    print("=" * 62)
    print("  Generating Physics-Aware EKF Baseline")
    print("  Pipeline: Raw Data -> Physics LSTM -> EKF Fusion -> CSV")
    print("=" * 62)

    df = pd.read_csv(DATA_PATH)
    all_runs = sorted(df['run_id'].unique())
    
    # Load the physics-aware LSTM bridge
    bridge = LSTMBridge(MODEL_PATH, STATS_PATH, device='cpu')
    
    results = []
    for run_id in all_runs:
        print(f"\n  Processing Run {run_id}...")
        df_run = df[df['run_id'] == run_id].sort_values('timestamp').reset_index(drop=True)
        
        ekf = AdaptiveEKF()
        # Run with the physics LSTM active (corrects wz + ax + inflates Q)
        df_out = run_ekf_on_run(df_run, ekf, bridge=bridge, use_lstm=True)
        df_out['run_id'] = run_id
        
        # Merge back the raw IMU columns we need for the locator
        df_merged = pd.merge(
            df_run[['timestamp', 'run_id', 'ax_corr', 'ay_corr', 'wz', 
                     'gt_heading', 'gt_speed_mps', 'pitch_deg', 'roll_deg']],
            df_out,
            on=['timestamp', 'run_id'],
            how='inner'
        )
        results.append(df_merged)

    final_df = pd.concat(results, ignore_index=True)
    
    # Calculate the ERROR targets for the LSTM locator
    final_df['error_x'] = final_df['gt_x'] - final_df['ekf_x']
    final_df['error_y'] = final_df['gt_y'] - final_df['ekf_y']
    
    final_df.to_csv(OUT_PATH, index=False)
    print(f"\n  DONE. Physics baseline saved to {OUT_PATH}")
    print(f"  Total Rows: {len(final_df)}")
    print(f"  Columns: {final_df.columns.tolist()}")
    print("=" * 62)

if __name__ == '__main__':
    main()

