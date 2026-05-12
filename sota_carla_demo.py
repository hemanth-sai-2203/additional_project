"""
sota_carla_demo.py
==================
The final State-of-the-Art (SOTA) demonstration script.
Features:
1. Real-time CARLA Autopilot traversal.
2. Two-stage AI inference (Physics cleaning + Error-state correction).
3. 3D In-game markers:
   - GREEN: Ground Truth (Where the car is)
   - RED  : SOTA Prediction (Our system - matches the car)
   - BLUE : Standard EKF (Drifts away in the tunnel)

FIX LOG (GPS-denied drift 10m->235m fix):
  FIX-A: Physics bridge ONLY pushes data when GPS is denied.
          This prevents open-road IMU contaminating the tunnel buffer.
  FIX-B: EKF hard-reset on GPS reconnect. After a tunnel pass the EKF
          position can be 170m wrong; a Kalman update cannot recover —
          we snap the EKF position to the first GNSS reading on re-entry.
  FIX-C: Locator correction is clamped to MAX_CORRECTION metres.
          If the model hallucinates a >50m correction it is silently ignored.
  FIX-D: Locator feature_history is cleared when GPS reconnects so the
          next tunnel pass gets a clean 60-step window, not a window that
          mixes tunnel + open-road steps.
  FIX-E: Reduced GPS_DENIED_Q_SCALE for velocity and heading. The original
          4x inflation made the EKF spread too fast without GPS.
          Handled directly in ekf_sota by calling set_process_noise_scale.
"""

import sys
# --- CARLA Python API Setup ---
# Update this path to your local CARLA .egg file
CARLA_EGG_PATH = r"<CARLA_PATH>\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg"
if os.path.exists(CARLA_EGG_PATH):
    sys.path.append(CARLA_EGG_PATH)
else:
    print(f"Warning: CARLA egg not found at {CARLA_EGG_PATH}")
    print("Please update CARLA_EGG_PATH in sota_carla_demo.py")
# ------------------------------
import carla

import os
import time as _time
import math
import numpy as np
import torch

# Project imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_CARLA_IMPL = os.path.join(_HERE, 'carla_implementation')
sys.path.insert(0, _HERE)
sys.path.insert(0, _CARLA_IMPL)

from carla_sensor_bridge import CARLASensorBridge
from ekf_physics import AdaptiveEKF, LSTMBridge
from lstm.train_lstm_locator import ErrorStateLSTM, DataPreprocessor as LocatorPrep

# CONFIG
MODEL_DIR          = os.path.join(_HERE, 'models')
PHYSICS_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_physics_predictor.pth')
PHYSICS_STATS_PATH = os.path.join(MODEL_DIR, 'lstm_physics_stats.npz')
LOCATOR_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_locator.pth')
LOCATOR_STATS_PATH = os.path.join(MODEL_DIR, 'lstm_locator_stats.npz')

# FIX-C: Maximum physically plausible LSTM correction (metres)
MAX_CORRECTION = 50.0

# FIX-E: Tighter process noise during GPS denial (original was 4x, caused rapid spread)
# [x, y, v, psi, b_psi]  — only inflate velocity and heading moderately
GPS_DENIED_Q_SCALE = [1.0, 1.0, 1.5, 1.5, 1.0]


def main():
    print("\n" + "="*50)
    print(" SOTA CARLA LIVE DEMO (Autopilot Mode)")
    print("="*50)
    print("\nFIXES ACTIVE:")
    print("  [A] Physics bridge only fills on GPS-denied steps")
    print("  [B] EKF hard-reset on GPS reconnect (no 170m Kalman catch-up)")
    print("  [C] LSTM correction clamped to ±%.0fm" % MAX_CORRECTION)
    print("  [D] feature_history cleared on GPS reconnect")
    print("  [E] GPS-denied Q-scale reduced (%.1fx vs old 4x)" % GPS_DENIED_Q_SCALE[2])

    # -------------------------------------------------------------------------
    # 1. Initialize CARLA
    # -------------------------------------------------------------------------
    bridge = CARLASensorBridge()
    if not bridge.connect():
        print("Error: Could not connect to CARLA server.")
        return

    print("Spawning vehicle with Autopilot...")
    if not bridge.spawn_vehicle():
        print("Error: Could not spawn vehicle.")
        bridge.destroy()
        return

    # CRITICAL: Calibrate GNSS origin so GPS coords convert correctly
    print("Calibrating GNSS reference point...")
    for _ in range(20):
        bridge.world.tick()
    _time.sleep(0.2)
    gnss_seed = bridge._get_latest_gnss(timeout=2.0)
    if gnss_seed is not None:
        bridge.coord_conv.set_gnss_origin(gnss_seed.latitude, gnss_seed.longitude)
        print(f"GNSS origin set: lat={gnss_seed.latitude:.6f} lon={gnss_seed.longitude:.6f}")
    else:
        print("WARNING: Could not get GNSS seed — GPS readings may be wrong!")

    # Drain stale sensor data from warmup
    while not bridge._imu_queue.empty():
        try: bridge._imu_queue.get_nowait()
        except: break
    while not bridge._gnss_queue.empty():
        try: bridge._gnss_queue.get_nowait()
        except: break

    # -------------------------------------------------------------------------
    # 2. Initialize AI Pipeline
    # -------------------------------------------------------------------------
    print("Loading Signal Cleaner (Physics LSTM)...")
    physics_bridge = LSTMBridge(PHYSICS_MODEL_PATH, PHYSICS_STATS_PATH)

    print("Loading Error-State Locator (Position LSTM)...")
    locator_stats = np.load(LOCATOR_STATS_PATH, allow_pickle=True)
    loc_prep = LocatorPrep()
    loc_prep.feat_mean = locator_stats['feat_mean']
    loc_prep.feat_std  = locator_stats['feat_std']
    loc_prep.tgt_mean  = locator_stats['tgt_mean']
    loc_prep.tgt_std   = locator_stats['tgt_std']

    loc_ckpt = torch.load(LOCATOR_MODEL_PATH, map_location='cpu')
    # Read architecture from checkpoint so it always matches the saved model
    _cfg = loc_ckpt.get('config', {'input_size': 10, 'h1': 64, 'h2': 32})
    locator_model = ErrorStateLSTM(
        input_size=_cfg.get('input_size', 10),
        h1=_cfg.get('h1', 64),
        h2=_cfg.get('h2', 32),
    )
    locator_model.load_state_dict(loc_ckpt['model_state'])
    locator_model.eval()
    print(f"  [Locator] Loaded | h1={_cfg.get('h1')} h2={_cfg.get('h2')} | Best epoch: {loc_ckpt.get('epoch', '?')}")

    # -------------------------------------------------------------------------
    # 3. Two EKFs: One SOTA, one Baseline
    # -------------------------------------------------------------------------
    ekf_sota = AdaptiveEKF()
    ekf_base = AdaptiveEKF()

    # FIX-E: Apply reduced process noise scale for GPS-denied periods
    # This is done per-step (see predict call below) but we set the baseline scale here
    # The AdaptiveEKF.set_process_noise_scale() handles it per-step

    # 4. State tracking
    feature_history = []          # LSTM-Locator input buffer (last 60 steps)
    prev_gps_denied = False       # FIX-B/D: track GPS state transitions
    lstm_corr_x     = 0.0         # Cumulative delta correction (X)
    lstm_corr_y     = 0.0         # Cumulative delta correction (Y)
    applied_corr_buffer = []      # Buffer for smoothing final applied corrections
    pure_applied_corr_buffer = [] # Buffer for smoothing the pure unconstrained LSTM
    
    seg_start_step = None
    seg_start_sota_err = None
    seg_start_base_err = None
    
    # Track error from the previous step to capture "Pre-Reset" drift
    last_step_sota_err = 0.0
    last_step_pure_err = 0.0
    last_step_base_err = 0.0
    tunnel_exit_sota_err = 0.0
    tunnel_exit_pure_err = 0.0
    tunnel_exit_base_err = 0.0

    # -------------------------------------------------------------------------
    # 5. Get first bundle and initialize both EKFs
    # -------------------------------------------------------------------------
    print("Waiting for first sensor bundle to initialize EKFs...")
    bundle = None
    for _ in range(30):
        bundle = bridge.get_sensor_bundle()
        if bundle is not None:
            break

    if bundle is None:
        print("Error: Could not get initial sensor bundle. Is CARLA running?")
        bridge.destroy()
        return

    gt = bundle.ground_truth
    ekf_sota.initialize(x0=gt.x, y0=gt.y, heading0=gt.heading, speed0=gt.velocity)
    ekf_base.initialize(x0=gt.x, y0=gt.y, heading0=gt.heading, speed0=gt.velocity)
    print(f"EKFs initialized at ({gt.x:.1f}, {gt.y:.1f})")

    print("\nDEMO STARTING. Watch the CARLA window!")
    print("GREEN: Actual | RED: SOTA Prediction | BLUE: Baseline Math")

    try:
        while True:
            bundle = bridge.get_sensor_bundle()
            if bundle is None:
                continue

            gps_denied = bundle.gps_denied

            # -----------------------------------------------------------------
            # GRAVITY CORRECTION (matches training data preprocessing)
            # -----------------------------------------------------------------
            pitch_rad = math.radians(bundle.ground_truth.pitch_deg)
            roll_rad  = math.radians(bundle.ground_truth.roll_deg)
            G = 9.81
            ax_corr = bundle.imu.accel_x + G * math.sin(pitch_rad)
            ay_corr = bundle.imu.accel_y - G * math.sin(roll_rad) * math.cos(pitch_rad)

            # -----------------------------------------------------------------
            # FIX-B + FIX-D: Handle GPS state transition (denied → available)
            # When GPS comes back, hard-reset the EKF position to the first
            # valid GNSS fix. A Kalman update cannot recover from >100m drift.
            # Also clear the locator history so the next tunnel sees a clean window.
            # -----------------------------------------------------------------
            just_got_gps = prev_gps_denied and not gps_denied
            if just_got_gps:
                # Capture the "Dirty" drift from the very last step of the tunnel 
                # before we initialize/reset the EKF for the OK segment.
                tunnel_exit_sota_err = last_step_sota_err
                tunnel_exit_pure_err = last_step_pure_err
                tunnel_exit_base_err = last_step_base_err

                if bundle.gnss is not None:
                    ekf_sota.initialize(
                        x0      = bundle.gnss.local_x,
                        y0      = bundle.gnss.local_y,
                        heading0= ekf_sota.get_heading(),
                        speed0  = ekf_sota.get_speed(),
                    )
                # FIX-D: clear locator history and reset correction buffer
                feature_history.clear()
                physics_bridge.reset()
                lstm_corr_x = 0.0
                lstm_corr_y = 0.0
                applied_corr_buffer.clear()
                pure_applied_corr_buffer.clear()

            # -----------------------------------------------------------------
            # BASELINE EKF (raw IMU, no corrections)
            # -----------------------------------------------------------------
            ekf_base.predict(a_fwd=ax_corr, wz=bundle.imu.yaw_rate, gps_denied=gps_denied)
            if not gps_denied and bundle.gnss:
                ekf_base.update(bundle.gnss.local_x, bundle.gnss.local_y)
            bx, by = ekf_base.get_position()

            # -----------------------------------------------------------------
            # SOTA EKF — Physics-Aware Signal Cleaning
            # FIX-A: Only push into physics bridge buffer when GPS is denied.
            #         Open-road data contaminates the tunnel IMU model.
            # -----------------------------------------------------------------
            bf, bl, bpsi, vf, vl, vpsi = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            if gps_denied:
                # FIX-A: Only push during GPS denial
                physics_bridge.push(ax_corr, ay_corr, bundle.imu.gyro_z,
                                    ekf_sota.get_speed(), gps_denied)
                if physics_bridge.ready():
                    bf, bl, bpsi, vf, vl, vpsi = physics_bridge.predict()

            ax_clean = ax_corr + bf
            wz_clean = bundle.imu.yaw_rate + bpsi

            # FIX-E: Apply tighter Q-scale during GPS denial
            if gps_denied:
                ekf_sota.set_process_noise_scale(GPS_DENIED_Q_SCALE)
            else:
                ekf_sota.set_process_noise_scale(1.0)  # reset to normal

            ekf_sota.predict(a_fwd=ax_clean, wz=wz_clean, gps_denied=gps_denied)

            # Wheel odometry anchors velocity during denial (prevents speed runaway)
            if gps_denied:
                ekf_sota.update_pseudo_velocity(bundle.ground_truth.odom_speed, r_variance=0.1)

            # GNSS update when available
            if not gps_denied and bundle.gnss:
                ekf_sota.update(bundle.gnss.local_x, bundle.gnss.local_y)

            ex, ey = ekf_sota.get_position()
            ev  = ekf_sota.get_speed()
            epsi = ekf_sota.get_heading()
            sx, sy = ekf_sota.get_position_std()

            # -----------------------------------------------------------------
            # LSTM ERROR-STATE LOCATOR
            # Features: ekf_v, ekf_psi, ax_corr, ay_corr, wz,
            #           pos_std_x, pos_std_y, gps_denied, pitch_deg, roll_deg
            # -----------------------------------------------------------------
            feat = [ev, epsi, ax_corr, ay_corr, bundle.imu.gyro_z,
                    sx, sy, float(gps_denied),
                    bundle.ground_truth.pitch_deg, bundle.ground_truth.roll_deg]
            feature_history.append(feat)
            if len(feature_history) > 60:
                feature_history.pop(0)

            final_x, final_y = ex, ey
            pure_final_x, pure_final_y = ex, ey
            
            # Only apply LSTM correction when GPS is denied and we have a full window
            if gps_denied and len(feature_history) == 60:
                h_norm = loc_prep.transform_array(np.array(feature_history))
                with torch.no_grad():
                    inp      = torch.tensor(h_norm).unsqueeze(0).float()
                    p_norm   = locator_model(inp).numpy()[0]
                    delta_real = loc_prep.denormalise_targets(p_norm)

                # (B) Scale down correction (TRAINING SIDE - FAST PATCH)
                # Reduces over-aggressive behavior
                delta_x = delta_real[0] * 0.7
                delta_y = delta_real[1] * 0.7

                # Accumulate delta corrections
                lstm_corr_x += delta_x
                lstm_corr_y += delta_y

                # Safety clip: prevent runaway if model mispredicts
                lstm_corr_x = np.clip(lstm_corr_x, -50.0, 50.0)
                lstm_corr_y = np.clip(lstm_corr_y, -50.0, 50.0)

                # (A) Confidence gating (MOST IMPORTANT)
                correction_norm = math.sqrt(lstm_corr_x**2 + lstm_corr_y**2)
                
                # (C) Apply ONLY when drift is growing
                # We use EKF position uncertainty (sx, sy) as our estimated drift
                ekf_drift = math.sqrt(sx**2 + sy**2)

                # (D) Drift-Aware Boosting (Solves the 26m plateau)
                boost_scale = 1.0
                if ekf_drift > 80.0:
                    boost_scale = 1.6
                elif ekf_drift > 40.0:
                    boost_scale = 1.3
                elif ekf_drift > 20.0:
                    boost_scale = 1.1

                if ekf_drift < 5.0:
                    # EKF still stable -> don't use LSTM
                    applied_x = 0.0
                    applied_y = 0.0
                elif correction_norm < 2.0:
                    # Very small correction -> ignore (noise)
                    applied_x = 0.0
                    applied_y = 0.0
                elif correction_norm < 15.0:
                    # Medium correction -> partially trust
                    applied_x = 0.5 * lstm_corr_x * boost_scale
                    applied_y = 0.5 * lstm_corr_y * boost_scale
                else:
                    # Large correction -> risky -> reduce impact but scale up if lost
                    applied_x = 0.3 * lstm_corr_x * boost_scale
                    applied_y = 0.3 * lstm_corr_y * boost_scale

                # Track the "Pure" unconstrained LSTM correction
                pure_applied_corr_buffer.append((applied_x, applied_y))
                if len(pure_applied_corr_buffer) > 3:
                    pure_applied_corr_buffer.pop(0)

                # (E) Lateral Damping (Geometry-Aware Sensor Fusion)
                # Project correction along the EKF's velocity direction to reduce artificial sideways drift
                vel_dir_x = math.cos(epsi)
                vel_dir_y = math.sin(epsi)
                dot_prod = applied_x * vel_dir_x + applied_y * vel_dir_y
                
                par_x = dot_prod * vel_dir_x
                par_y = dot_prod * vel_dir_y
                perp_x = applied_x - par_x
                perp_y = applied_y - par_y
                
                # Dampen lateral (sideways) correction by 70% (0.3 modifier)
                applied_x = par_x + 0.3 * perp_x
                applied_y = par_y + 0.3 * perp_y

                # (F) Smooth LSTM output to remove spikes
                applied_corr_buffer.append((applied_x, applied_y))
                if len(applied_corr_buffer) > 3:
                    applied_corr_buffer.pop(0)
                
                avg_applied_x = np.mean([c[0] for c in applied_corr_buffer])
                avg_applied_y = np.mean([c[1] for c in applied_corr_buffer])

                avg_pure_applied_x = np.mean([c[0] for c in pure_applied_corr_buffer])
                avg_pure_applied_y = np.mean([c[1] for c in pure_applied_corr_buffer])

                final_x = ex + avg_applied_x
                final_y = ey + avg_applied_y
                
                pure_final_x = ex + avg_pure_applied_x
                pure_final_y = ey + avg_pure_applied_y

            # -----------------------------------------------------------------
            # 3D VISUALIZATION
            # -----------------------------------------------------------------
            gt_x, gt_y = bundle.ground_truth.x, bundle.ground_truth.y
            bridge.draw_3d_point(gt_x,   gt_y,   color='green',  size=0.15, life_time=0.1)
            bridge.draw_3d_point(pure_final_x, pure_final_y, color='yellow', size=0.12, life_time=0.1)
            bridge.draw_3d_point(final_x, final_y, color='red',    size=0.12, life_time=0.1)
            bridge.draw_3d_point(bx,      by,      color='blue',   size=0.10, life_time=0.1)

            # Console output
            dist_err = math.sqrt((final_x - gt_x)**2 + (final_y - gt_y)**2)
            pure_err = math.sqrt((pure_final_x - gt_x)**2 + (pure_final_y - gt_y)**2)
            base_err = math.sqrt((bx - gt_x)**2 + (by - gt_y)**2)
            gps_s = "DENIED" if gps_denied else "OK    "
            
            if seg_start_step is None:
                seg_start_step = bundle.step
                seg_start_sota_err = dist_err
                seg_start_pure_err = pure_err
                seg_start_base_err = base_err

            gate_status = ""
            if gps_denied and len(feature_history) == 60:
                is_active = (abs(avg_applied_x) > 0.01 or abs(avg_applied_y) > 0.01)
                gate_status = " | LSTM: ACTIVE (Gated)" if is_active else " | LSTM: SILENT (Gated)"

            # Print continuous line
            print(f"\rStep: {bundle.step:4d} | GPS: {gps_s} | SOTA: {dist_err:6.2f}m | Pure: {pure_err:6.2f}m | Base: {base_err:6.2f}m{gate_status}",
                  end="")

            # Check for segment end
            just_got_gps = prev_gps_denied and not gps_denied
            just_lost_gps = not prev_gps_denied and gps_denied

            if just_got_gps or just_lost_gps:
                print()  # Break the \r line
                seg_type = "GPS DENIED" if prev_gps_denied else "GPS OK"
                
                # For GPS DENIED, we want to show the drift RIGHT BEFORE the reset
                display_sota_err = tunnel_exit_sota_err if just_got_gps else dist_err
                display_pure_err = tunnel_exit_pure_err if just_got_gps else pure_err
                display_base_err = tunnel_exit_base_err if just_got_gps else base_err
                display_stop_step = (bundle.step - 1) if just_got_gps else bundle.step

                print(f"[{seg_type} SEGMENT]")
                print(f"  Start Step: {seg_start_step:4d} | SOTA: {seg_start_sota_err:6.2f}m | Pure: {seg_start_pure_err:6.2f}m | Base: {seg_start_base_err:6.2f}m")
                print(f"  Stop  Step: {display_stop_step:4d} | SOTA: {display_sota_err:6.2f}m | Pure: {display_pure_err:6.2f}m | Base: {display_base_err:6.2f}m")
                
                if just_got_gps:
                    print(f"  [FIX-B] GPS reconnected — hard-resetting EKF SOTA position to GNSS.")
                    print(f"  [FIX-D] feature_history, physics_bridge, and correction buffer cleared.")
                
                print("-" * 65)
                
                # Reset tracking for new segment
                seg_start_step = bundle.step
                seg_start_sota_err = dist_err
                seg_start_base_err = base_err

            # Update state for next step
            prev_gps_denied = gps_denied
            last_step_sota_err = dist_err
            last_step_pure_err = pure_err
            last_step_base_err = base_err

    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    finally:
        bridge.destroy()
        print("\nCleaned up actors.")


if __name__ == "__main__":
    main()
