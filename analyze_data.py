import pandas as pd
import numpy as np

df = pd.read_csv('data/town04_physics_baseline_v2.csv')
print('Columns:', list(df.columns))
print('Shape:', df.shape)
print('Runs:', sorted(df['run_id'].unique()))

print('\nPer-run tunnel stats:')
for r in sorted(df['run_id'].unique()):
    dr = df[df.run_id == r]
    tun = dr[dr['gps_denied'] > 0.5]
    print(f'  Run {r}: total={len(dr)}, tunnel={len(tun)}')

print('\nError stats in tunnel (per run):')
for r in sorted(df['run_id'].unique()):
    tun = df[(df.run_id == r) & (df['gps_denied'] > 0.5)]
    if len(tun) > 0:
        err = np.sqrt(tun['error_x']**2 + tun['error_y']**2)
        print(f'  Run {r}: mean_err={err.mean():.2f}, max_err={err.max():.2f}, '
              f'err_x range=[{tun["error_x"].min():.2f}, {tun["error_x"].max():.2f}], '
              f'err_y range=[{tun["error_y"].min():.2f}, {tun["error_y"].max():.2f}]')

# Analyze tunnel segment lengths
print('\nTunnel segment analysis:')
for r in sorted(df['run_id'].unique()):
    dr = df[df.run_id == r].reset_index(drop=True)
    gps = dr['gps_denied'].values
    segments = []
    in_tunnel = False
    start = 0
    for i in range(len(gps)):
        if gps[i] > 0.5 and not in_tunnel:
            start = i
            in_tunnel = True
        elif gps[i] < 0.5 and in_tunnel:
            segments.append(i - start)
            in_tunnel = False
    if in_tunnel:
        segments.append(len(gps) - start)
    if segments:
        print(f'  Run {r}: {len(segments)} tunnel segments, lengths: {segments}, '
              f'min={min(segments)}, max={max(segments)}, mean={np.mean(segments):.0f}')

# Check the delta error distribution
print('\nDelta error stats (what the LSTM learns):')
for r in sorted(df['run_id'].unique()):
    dr = df[df.run_id == r].reset_index(drop=True)
    tun = dr[dr['gps_denied'] > 0.5]
    if len(tun) > 1:
        delta_x = tun['error_x'].diff().dropna()
        delta_y = tun['error_y'].diff().dropna()
        print(f'  Run {r}: delta_x: mean={delta_x.mean():.4f}, std={delta_x.std():.4f}, '
              f'delta_y: mean={delta_y.mean():.4f}, std={delta_y.std():.4f}')

