"""
collect_data.py  —  CARLA Town04 Data Collection  (v10 — Wheel Odometry)
=========================================================================
For: LSTM + RL-Adaptive EKF Localization Project

What changed from v9 → v10
--------------------------
  1. WHEEL ODOMETRY (biggest addition — data_requirements.md §1)
     - odom_speed_mps: speed from wheel rotation, NOT integrated IMU
     - wheel_fl_rpm, wheel_fr_rpm, wheel_rl_rpm, wheel_rr_rpm
     - Why: IMU tunnel drift = integrating a_fwd over time. Wheel odometry
       gives direct speed with NO integration drift → ~60-80% RMSE drop.
     - Wheel radius read from vehicle.get_physics_control() at spawn.
     - RPM = (speed / circumference) × 60 with independent Gaussian noise
       per wheel (0.5% reading noise, floor 0.02 m/s).

  2. WEATHER VARIETY — 6 presets up from 4 (data_requirements.md §2)
     - Added: HardRainNoon (heavy GPS degradation) + ClearNight
     - GNSS noise stddev varies per weather (rain = 2× worse GPS)
     - NUM_RUNS = 6 (one per weather preset = 30 min total)

  3. pitch_deg, roll_deg already in TRAIN_COLS — no change needed.
     (data_requirements.md §4: "free upgrade, already collected")

All v9 crash-safety fixes preserved (FIX 1–5).
"""

import sys
import os
import time
import math
import csv
import queue
import random
import numpy as np
from scipy.signal import filtfilt, butter

# =============================================================================
# CARLA IMPORT
# =============================================================================
CARLA_EGG = (
    r'C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor'
    r'\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg'
)
sys.path.insert(0, CARLA_EGG)
sys.path.insert(0, r'C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor\PythonAPI')
import carla


# =============================================================================
# CONFIGURATION
# =============================================================================
HOST      = '127.0.0.1'
PORT      = 2000
TIMEOUT_S = 15.0
MAP_NAME  = 'Town04'

VEHICLE_BP    = 'vehicle.tesla.model3'
FIXED_DELTA_T = 0.05        # 20 Hz
WARMUP_TICKS  = 120
TICKS_PER_RUN = 10_000       # 5 min per run
NUM_RUNS      = 6            # one per weather preset
SAVE_INTERVAL = 500
NPC_COUNT     = 30

G = 9.81
EGO_EXCLUSION_RADIUS = 10.0

HIGHWAY_SPAWN_INDICES = [16, 16, 17, 16, 15, 16] # 6 entries

TUNNEL_X_MIN = -130.0
TUNNEL_X_MAX =  140.0
TUNNEL_Y_MIN =  -35.0
TUNNEL_Y_MAX =   65.0

# 6 weather presets — one per run
WEATHER_PRESETS = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.ClearSunset,
    carla.WeatherParameters.ClearNight,
]
WEATHER_NAMES = [
    "ClearNoon", "CloudyNoon", "WetNoon",
    "HardRainNoon", "ClearSunset", "ClearNight",
]

# GPS noise degrades in rain — realistic model
GNSS_NOISE_BY_WEATHER = {
    "ClearNoon":    "0.000009",   # ~1.0 m
    "CloudyNoon":   "0.000009",   # ~1.0 m
    "WetNoon":      "0.000013",   # ~1.4 m
    "HardRainNoon": "0.000018",   # ~2.0 m  ← significant degradation
    "ClearSunset":  "0.000009",   # ~1.0 m
    "ClearNight":   "0.000011",   # ~1.2 m
}

# Wheel encoder noise model
WHEEL_NOISE_SIGMA_FRAC = 0.005   # 0.5% of reading
WHEEL_NOISE_FLOOR_MPS  = 0.02    # minimum noise (m/s equivalent)
DEFAULT_WHEEL_RADIUS_M = 0.338   # Tesla Model 3 fallback

OUTPUT_DIR = r'C:\Users\heman\Music\rl_imu_project\data'
TRAIN_CSV  = os.path.join(OUTPUT_DIR, 'town04_dataset.csv')
DEBUG_CSV  = os.path.join(OUTPUT_DIR, 'town04_debug.csv')

# All training columns — including v10 wheel odometry columns
TRAIN_COLS = [
    'timestamp', 'run_id', 'weather',
    'ax', 'ay', 'az', 'wx', 'wy', 'wz',
    'ax_corr', 'ay_corr',
    'gnss_x', 'gnss_y',
    'gt_x', 'gt_y', 'gt_heading', 'gt_speed_mps',
    'gt_accel_fwd_mps2', 'gt_accel_lat_mps2',
    'gps_denied', 'pitch_deg', 'roll_deg',
    # v10 NEW: wheel odometry
    'odom_speed_mps',
    'wheel_fl_rpm', 'wheel_fr_rpm', 'wheel_rl_rpm', 'wheel_rr_rpm',
]

DEBUG_COLS = TRAIN_COLS + [
    'world_x', 'world_y', 'world_z',
    'gnss_x_raw', 'gnss_y_raw',
]

SPEED_SCHEDULE = [
    (100, 60,  "highway_cruise"), (80,  80,  "fast_highway"),
    (60,  90,  "max_highway"),    (80,  40,  "decelerate"),
    (50,   0,  "full_stop"),      (40,   0,  "stopped"),
    (80,  20,  "slow_creep"),     (100, 50,  "moderate_urban"),
    (150, 60,  "highway_cruise"), (60,  30,  "slow_urban"),
    (100, 60,  "highway_cruise"), (80,  70,  "slightly_above"),
    (100, 60,  "highway_cruise"), (80,   0,  "emergency_stop"),
    (40,   0,  "stopped_2"),      (100, 60,  "resume_highway"),
    (100, 60,  "highway_cruise"),
]

FILTFILT_MIN_SAMPLES = 16


# =============================================================================
# ZERO-PHASE FILTER
# =============================================================================
def apply_zero_phase_filter(data_array, cutoff_freq=2.0, fs=20.0, order=2):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff_freq / nyquist, btype='low', analog=False)
    return filtfilt(b, a, data_array)


def safe_filter_array(arr):
    if np.isnan(arr).all():
        return arr.copy()
    arr = arr.copy()
    mask = np.isnan(arr)
    if mask.any():
        valid = ~mask
        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(valid), arr[valid])
    return apply_zero_phase_filter(arr)


# =============================================================================
# SENSOR MANAGER & COORD CONVERTER
# =============================================================================
class SyncSensorManager:
    def __init__(self):
        self.imu_queue  = queue.Queue()
        self.gnss_queue = queue.Queue()

    def on_imu(self, data):  self.imu_queue.put(data)
    def on_gnss(self, data): self.gnss_queue.put(data)

    def get_frame(self, frame_id, timeout=2.0):
        return (self._drain_to_frame(self.imu_queue,  frame_id, timeout),
                self._drain_to_frame(self.gnss_queue, frame_id, timeout))

    def _drain_to_frame(self, q, target_frame, timeout):
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError(f"Timeout waiting for frame {target_frame}")
            try:
                data = q.get(timeout=min(remaining, 0.5))
                if data.frame == target_frame:
                    return data
            except queue.Empty:
                continue

    def clear(self):
        for q in (self.imu_queue, self.gnss_queue):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break


class CoordConverter:
    EARTH_RADIUS = 6_371_000

    def __init__(self):
        self.ref_lat = self.ref_lon = None

    def set_origin(self, lat, lon):
        self.ref_lat, self.ref_lon = lat, lon
        print(f"  [Coord] Origin: lat={lat:.8f}  lon={lon:.8f}")

    def _check_origin(self):
        if self.ref_lat is None:
            raise RuntimeError("Origin not set — call set_origin() first.")

    def gnss_to_local(self, lat, lon):
        self._check_origin()
        e = self.EARTH_RADIUS * math.radians(lon - self.ref_lon) * math.cos(math.radians(self.ref_lat))
        n = self.EARTH_RADIUS * math.radians(lat - self.ref_lat)
        return e, -n   # CARLA +Y is South

    def gnss_to_local_raw(self, lat, lon):
        self._check_origin()
        e = self.EARTH_RADIUS * math.radians(lon - self.ref_lon) * math.cos(math.radians(self.ref_lat))
        n = self.EARTH_RADIUS * math.radians(lat - self.ref_lat)
        return e, n


def correct_imu_for_gravity(ax_raw, ay_raw, pitch_deg, roll_deg):
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    return round(ax_raw + G * math.sin(p), 6), \
           round(ay_raw - G * math.sin(r) * math.cos(p), 6)


# =============================================================================
# WHEEL ODOMETRY  (v10 new)
# =============================================================================
def get_wheel_radius(vehicle):
    """
    Read wheel radius from CARLA physics control.
    Returns metres. Falls back to DEFAULT_WHEEL_RADIUS_M if API returns 0.
    """
    try:
        phys = vehicle.get_physics_control()
        r_cm = phys.wheels[0].radius
        if r_cm > 0:
            return r_cm / 100.0
    except Exception:
        pass
    return DEFAULT_WHEEL_RADIUS_M


def compute_wheel_odometry(gt_speed_mps, wheel_radius_m):
    """
    Simulates 4 independent wheel encoders.

    Physics:
        RPM_true = (speed_mps / (2π × radius_m)) × 60

    Noise: each wheel has independent Gaussian noise in the speed domain,
    then converted to RPM. This simulates real encoder imperfection.

    Returns: (odom_speed_mps, fl_rpm, fr_rpm, rl_rpm, rr_rpm)
    """
    circ  = 2.0 * math.pi * wheel_radius_m
    if circ <= 0:
        circ = 2.0 * math.pi * DEFAULT_WHEEL_RADIUS_M

    true_rpm = (gt_speed_mps / circ) * 60.0

    # Noise sigma in m/s, then convert to RPM
    sigma_mps = max(WHEEL_NOISE_FLOOR_MPS, WHEEL_NOISE_SIGMA_FRAC * gt_speed_mps)
    sigma_rpm = (sigma_mps / circ) * 60.0

    fl = max(0.0, true_rpm + np.random.normal(0.0, sigma_rpm))
    fr = max(0.0, true_rpm + np.random.normal(0.0, sigma_rpm))
    rl = max(0.0, true_rpm + np.random.normal(0.0, sigma_rpm))
    rr = max(0.0, true_rpm + np.random.normal(0.0, sigma_rpm))

    odom_speed = ((fl + fr + rl + rr) / 4.0 / 60.0) * circ

    return (round(odom_speed, 5),
            round(fl, 3), round(fr, 3),
            round(rl, 3), round(rr, 3))


# =============================================================================
# HELPERS
# =============================================================================
class SpeedScheduler:
    def __init__(self, schedule):
        self.schedule = schedule
        self.total = sum(d for d, _, _ in schedule)

    def get(self, tick):
        pos, cum = tick % self.total, 0
        for dur, spd, lbl in self.schedule:
            cum += dur
            if pos < cum:
                return spd, lbl
        return self.schedule[-1][1], self.schedule[-1][2]


def get_speed_mps(v):
    vel = v.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


def in_tunnel(x, y):
    return TUNNEL_X_MIN <= x <= TUNNEL_X_MAX and TUNNEL_Y_MIN <= y <= TUNNEL_Y_MAX


def spectator_follow(world, vehicle):
    t   = vehicle.get_transform()
    yaw = math.radians(t.rotation.yaw)
    world.get_spectator().set_transform(carla.Transform(
        carla.Location(x=t.location.x - 10*math.cos(yaw),
                       y=t.location.y - 10*math.sin(yaw),
                       z=t.location.z + 6),
        carla.Rotation(pitch=-20, yaw=t.rotation.yaw),
    ))


def carla_yaw_to_heading_rad(yaw_deg):
    return math.atan2(math.sin(-math.radians(yaw_deg)),
                      math.cos(-math.radians(yaw_deg)))


def spawn_npcs(world, count, ego_loc):
    bpl = world.get_blueprint_library()
    safe_pts = [
        sp for sp in world.get_map().get_spawn_points()
        if math.sqrt((sp.location.x - ego_loc.x)**2 +
                     (sp.location.y - ego_loc.y)**2) > EGO_EXCLUSION_RADIUS
    ]
    random.shuffle(safe_pts)
    bps = [bp for bp in bpl.filter('vehicle.*')
           if not any(x in bp.id for x in ['bike', 'motorcycle', 'crossbike'])]
    npcs = []
    for sp in safe_pts[:count]:
        npc = world.try_spawn_actor(random.choice(bps), sp)
        if npc:
            npc.set_autopilot(True, 8000)
            npcs.append(npc)
    print(f"  [NPC] Spawned {len(npcs)} vehicles.")
    return npcs


def destroy_npcs(client, npcs):
    if npcs:
        client.apply_batch([carla.command.DestroyActor(n) for n in npcs])
    print(f"  [NPC] Destroyed {len(npcs)} vehicles.")


def verify_alignment(buffer):
    print("\n" + "="*72)
    print("  ALIGNMENT VERIFICATION (First 20 valid ticks)")
    print("="*72)
    print(f"  {'t':>3}  {'gt_x':>8}  {'gt_y':>8}  {'gnss_x':>8}  {'gnss_y':>8}  "
          f"{'err_x':>7}  {'err_y':>7}")
    errs_x, errs_y = [], []
    for i, r in enumerate(buffer):
        dx, dy = r['gnss_x'] - r['gt_x'], r['gnss_y'] - r['gt_y']
        errs_x.append(abs(dx)); errs_y.append(abs(dy))
        print(f"  {i:>3}  {r['gt_x']:>8.3f}  {r['gt_y']:>8.3f}  "
              f"{r['gnss_x']:>8.3f}  {r['gnss_y']:>8.3f}  {dx:>7.3f}  {dy:>7.3f}")
    if len(buffer) >= 2:
        gt_dy  = buffer[-1]['gt_y']   - buffer[0]['gt_y']
        gns_dy = buffer[-1]['gnss_y'] - buffer[0]['gnss_y']
        if gt_dy * gns_dy < -0.1:
            print("\n  CRITICAL: gt_y and gnss_y move in OPPOSITE directions!\n")
        else:
            print(f"\n  OK | Mean Err X={np.mean(errs_x):.2f}m  Y={np.mean(errs_y):.2f}m\n")


# =============================================================================
# SINGLE RUN COLLECTION
# =============================================================================
def collect_run(client, world, tm, run_id, spawn_index,
                weather_preset, weather_name,
                conv, scheduler, tw, dw, train_f, debug_f):
    vehicle = imu_s = gnss_s = None
    npcs               = []
    run_rows           = []
    align_buf          = []
    raw_accel_fwd_list = []
    raw_accel_lat_list = []
    run_exception      = None
    wheel_radius_m     = DEFAULT_WHEEL_RADIUS_M

    temp_train_path = os.path.join(OUTPUT_DIR, f'temp_train_run_{run_id}.csv')
    temp_debug_path = os.path.join(OUTPUT_DIR, f'temp_debug_run_{run_id}.csv')

    try:
        temp_train_f = open(temp_train_path, 'w', newline='', encoding='utf-8')
        temp_debug_f = open(temp_debug_path, 'w', newline='', encoding='utf-8')
        temp_tw = csv.DictWriter(temp_train_f, fieldnames=TRAIN_COLS)
        temp_dw = csv.DictWriter(temp_debug_f, fieldnames=DEBUG_COLS)
        temp_tw.writeheader(); temp_dw.writeheader()

        world.set_weather(weather_preset)
        print(f"\n[Run {run_id}] Weather: {weather_name}")

        bpl = world.get_blueprint_library()
        bp  = bpl.find(VEHICLE_BP)
        bp.set_attribute('role_name', 'hero')
        spawn_pts = world.get_map().get_spawn_points()

        sp      = spawn_pts[spawn_index]
        vehicle = world.try_spawn_actor(bp, sp)
        if not vehicle:
            for alt in [spawn_index-1, spawn_index+1, spawn_index-2]:
                if 0 <= alt < len(spawn_pts):
                    vehicle = world.try_spawn_actor(bp, spawn_pts[alt])
                    if vehicle:
                        sp = spawn_pts[alt]; break
        if not vehicle:
            raise RuntimeError("Vehicle spawn failed.")

        world.tick()

        # Read wheel radius from physics
        wheel_radius_m = get_wheel_radius(vehicle)
        circ_cm = 2 * math.pi * wheel_radius_m * 100
        print(f"  [Odom] Wheel radius={wheel_radius_m*100:.1f}cm  "
              f"circumference={circ_cm:.1f}cm")

        sync_mgr = SyncSensorManager()

        # IMU sensor
        imu_bp = bpl.find('sensor.other.imu')
        for k, v in [
            ('noise_accel_stddev_x','0.02'), ('noise_accel_stddev_y','0.02'),
            ('noise_accel_stddev_z','0.02'),
            ('noise_gyro_stddev_x', '0.005'),('noise_gyro_stddev_y', '0.005'),
            ('noise_gyro_stddev_z', '0.005'),
            ('noise_gyro_bias_x',   '0.001'),('noise_gyro_bias_y',   '0.001'),
            ('noise_gyro_bias_z',   '0.001'),
            ('sensor_tick', str(FIXED_DELTA_T)),
        ]:
            imu_bp.set_attribute(k, v)
        imu_s = world.spawn_actor(imu_bp,
            carla.Transform(carla.Location(x=0,y=0,z=0)), attach_to=vehicle)
        imu_s.listen(sync_mgr.on_imu)

        # GNSS sensor — noise varies per weather
        gnss_noise = GNSS_NOISE_BY_WEATHER.get(weather_name, "0.000009")
        gnss_bp = bpl.find('sensor.other.gnss')
        for k, v in [
            ('noise_lat_stddev', gnss_noise),
            ('noise_lon_stddev', gnss_noise),
            ('sensor_tick', str(FIXED_DELTA_T)),
        ]:
            gnss_bp.set_attribute(k, v)
        gnss_s = world.spawn_actor(gnss_bp,
            carla.Transform(carla.Location(x=0,y=0,z=0)), attach_to=vehicle)
        gnss_s.listen(sync_mgr.on_gnss)
        print(f"  [GNSS] noise_stddev={gnss_noise} ({weather_name})")

        tm.ignore_lights_percentage(vehicle, 0.0)
        tm.ignore_signs_percentage(vehicle, 0.0)
        tm.auto_lane_change(vehicle, True)
        vehicle.set_autopilot(True, 8000)

        npcs = spawn_npcs(world, NPC_COUNT, sp.location)

        print(f"  Warming up ({WARMUP_TICKS} ticks)...")
        for _ in range(WARMUP_TICKS):
            frame = world.tick()
            try: sync_mgr.get_frame(frame, timeout=1.0)
            except RuntimeError: pass

        print("  Setting GNSS origin...")
        origin_set = False
        for _ in range(30):
            frame = world.tick()
            try:
                _, gnss_data = sync_mgr.get_frame(frame, timeout=5.0)
                conv.set_origin(gnss_data.latitude, gnss_data.longitude)
                origin_set = True; break
            except RuntimeError:
                continue
        if not origin_set:
            raise RuntimeError("GNSS not ready after 30 ticks.")

        spawn_loc = vehicle.get_transform().location
        sync_mgr.clear()

        print(f"  Recording {TICKS_PER_RUN} ticks...")
        tick = tunnel_ticks = 0

        # Initialize once (before loop)
        burst_end = -1

        # Initialize once (before loop)
        prev_gnss_x, prev_gnss_y = 0.0, 0.0
        while tick < TICKS_PER_RUN:
            frame_id = world.tick()
            tick += 1
            target_kmh, phase = scheduler.get(tick)
            tm.set_desired_speed(vehicle, float(target_kmh))

            # Dropped-frame path
            try:
                imu_data, gnss_data = sync_mgr.get_frame(frame_id)
            except RuntimeError as e:
                print(f"  [WARN] tick {tick}: dropped ({e})")
                ts    = round(tick * FIXED_DELTA_T, 4)
                t_row = {c: float('nan') for c in TRAIN_COLS}
                t_row.update({'timestamp': ts, 'run_id': run_id, 'weather': weather_name})
                d_row = {c: float('nan') for c in DEBUG_COLS}
                d_row.update({'timestamp': ts, 'run_id': run_id, 'weather': weather_name})
                temp_tw.writerow(t_row); temp_dw.writerow(d_row)
                run_rows.append((t_row, d_row))
                raw_accel_fwd_list.append(float('nan'))
                raw_accel_lat_list.append(float('nan'))
                continue

            # Normal path
            transform = vehicle.get_transform()
            loc       = transform.location
            pitch_deg = transform.rotation.pitch
            roll_deg  = transform.rotation.roll
            yaw_deg   = transform.rotation.yaw

            gt_x      = loc.x - spawn_loc.x
            gt_y      = loc.y - spawn_loc.y
            gt_heading = carla_yaw_to_heading_rad(yaw_deg)
            gt_speed   = get_speed_mps(vehicle)

            accel_3d = vehicle.get_acceleration()
            fwd_vec  = transform.get_forward_vector()
            lat_vec  = transform.get_right_vector()
            gt_accel_fwd = max(-25.0, min(25.0,
                accel_3d.x*fwd_vec.x + accel_3d.y*fwd_vec.y + accel_3d.z*fwd_vec.z))
            gt_accel_lat = max(-25.0, min(25.0,
                accel_3d.x*lat_vec.x + accel_3d.y*lat_vec.y + accel_3d.z*lat_vec.z))

            gnss_x, gnss_y         = conv.gnss_to_local(gnss_data.latitude, gnss_data.longitude)
            gnss_x_raw, gnss_y_raw = conv.gnss_to_local_raw(gnss_data.latitude, gnss_data.longitude)
            ax_corr, ay_corr = correct_imu_for_gravity(
                imu_data.accelerometer.x, imu_data.accelerometer.y,
                pitch_deg, roll_deg)

            # Base: real tunnel condition
            gps_denied = 1 if in_tunnel(loc.x, loc.y) else 0

         

            # Random short dropouts (LOW probability)
            if not gps_denied and random.random() < 0.02:
                gps_denied = 1

            # Controlled bursts (only if NOT already active)
            if burst_end < tick and random.random() < 0.005:
                burst_length = random.randint(50, 150)
                burst_end = tick + burst_length

            if tick < burst_end:
                gps_denied = 1

            # Count tunnel-only stats (optional but clean)
            if in_tunnel(loc.x, loc.y):
                tunnel_ticks += 1

            if gps_denied:
                gnss_x, gnss_y = prev_gnss_x, prev_gnss_y
            else:
                prev_gnss_x, prev_gnss_y = gnss_x, gnss_y

            odom_spd, fl_rpm, fr_rpm, rl_rpm, rr_rpm = compute_wheel_odometry(
                gt_speed, wheel_radius_m)

            ts = round(tick * FIXED_DELTA_T, 4)

            t_row = {
                'timestamp': ts, 'run_id': run_id, 'weather': weather_name,
                'ax':  round(imu_data.accelerometer.x, 6),
                'ay':  round(imu_data.accelerometer.y, 6),
                'az':  round(imu_data.accelerometer.z, 6),
                'wx':  round(imu_data.gyroscope.x, 6),
                'wy':  round(imu_data.gyroscope.y, 6),
                'wz':  round(imu_data.gyroscope.z, 6),
                'ax_corr': ax_corr, 'ay_corr': ay_corr,
                'gnss_x':  round(gnss_x, 4), 'gnss_y': round(gnss_y, 4),
                'gt_x':    round(gt_x, 4),   'gt_y':   round(gt_y, 4),
                'gt_heading':        round(gt_heading, 6),
                'gt_speed_mps':      round(gt_speed, 4),
                'gt_accel_fwd_mps2': round(gt_accel_fwd, 4),
                'gt_accel_lat_mps2': round(gt_accel_lat, 4),
                'gps_denied': gps_denied,
                'pitch_deg':  round(pitch_deg, 3),
                'roll_deg':   round(roll_deg, 3),
                # v10 wheel odometry
                'odom_speed_mps': odom_spd,
                'wheel_fl_rpm':   fl_rpm, 'wheel_fr_rpm': fr_rpm,
                'wheel_rl_rpm':   rl_rpm, 'wheel_rr_rpm': rr_rpm,
            }
            d_row = {
                **t_row,
                'world_x': round(loc.x, 4), 'world_y': round(loc.y, 4),
                'world_z': round(loc.z, 4),
                'gnss_x_raw': round(gnss_x_raw, 4),
                'gnss_y_raw': round(gnss_y_raw, 4),
            }

            temp_tw.writerow(t_row); temp_dw.writerow(d_row)
            run_rows.append((t_row, d_row))
            raw_accel_fwd_list.append(gt_accel_fwd)
            raw_accel_lat_list.append(gt_accel_lat)

            if len(align_buf) < 20 and not math.isnan(gnss_x_raw):
                align_buf.append(t_row)

            if tick % SAVE_INTERVAL == 0:
                temp_train_f.flush(); temp_debug_f.flush()
            if tick % 5 == 0:
                spectator_follow(world, vehicle)
            if tick % 500 == 0:
                print(f"  tick={tick:5d}/{TICKS_PER_RUN} "
                      f"| spd={gt_speed*3.6:5.1f}km/h "
                      f"| odom={odom_spd*3.6:5.1f}km/h "
                      f"| GPS={'DENY' if gps_denied else 'OK  '} "
                      f"| {phase}")

        if run_id == 0 and align_buf:
            verify_alignment(align_buf)
        print(f"  [Run {run_id}] Tunnel: {tunnel_ticks}/{TICKS_PER_RUN} "
              f"({100*tunnel_ticks/TICKS_PER_RUN:.1f}%)")
        
        denied_count = sum(1 for r,_ in run_rows if r['gps_denied'] == 1)
        print(f"  GPS Denied: {denied_count}/{len(run_rows)} ({100*denied_count/len(run_rows):.1f}%)")

    except Exception as e:
        run_exception = e
        print(f"  [ERROR] Run {run_id}: {e}")

    finally:
        for fname in ('temp_train_f', 'temp_debug_f'):
            try:
                h = locals().get(fname)
                if h: h.close()
            except Exception: pass
        for sensor in [imu_s, gnss_s]:
            try:
                if sensor: sensor.stop(); sensor.destroy()
            except Exception: pass
        try:
            if vehicle: vehicle.destroy()
        except Exception: pass
        destroy_npcs(client, npcs)
        for _ in range(15):
            try: world.tick()
            except Exception: continue

    # Post-run: filter + write to master CSV
    if run_rows:
        n = len(run_rows)
        print(f"  [Run {run_id}] {n} rows collected.")
        if n < FILTFILT_MIN_SAMPLES:
            print(f"  [Run {run_id}] Too short — discarding.")
        else:
            print(f"  [Run {run_id}] Filtering and writing to master CSV...")
            fwd_arr = np.array(raw_accel_fwd_list)
            lat_arr = np.array(raw_accel_lat_list)
            orig_fwd_nan = np.isnan(fwd_arr)
            orig_lat_nan = np.isnan(lat_arr)
            fwd_sm = safe_filter_array(fwd_arr)
            lat_sm = safe_filter_array(lat_arr)
            for i, (t_row, d_row) in enumerate(run_rows):
                if not math.isnan(t_row['timestamp']):
                    if not orig_fwd_nan[i]:
                        t_row['gt_accel_fwd_mps2'] = round(fwd_sm[i], 4)
                        d_row['gt_accel_fwd_mps2'] = round(fwd_sm[i], 4)
                    if not orig_lat_nan[i]:
                        t_row['gt_accel_lat_mps2'] = round(lat_sm[i], 4)
                        d_row['gt_accel_lat_mps2'] = round(lat_sm[i], 4)
                tw.writerow(t_row); dw.writerow(d_row)
            print(f"  [Run {run_id}] Done.")
    else:
        print(f"  [Run {run_id}] No rows collected.")

    for path in (temp_train_path, temp_debug_path):
        try: os.remove(path)
        except OSError: pass

    if run_exception is not None:
        raise run_exception


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = orig_settings = train_f = debug_f = None

    print("\n" + "="*65)
    print("  CARLA Town04 — Data Collection  (v10 — Wheel Odometry)")
    print(f"  {NUM_RUNS} runs × {TICKS_PER_RUN * FIXED_DELTA_T / 60:.0f} min = "
          f"{NUM_RUNS * TICKS_PER_RUN * FIXED_DELTA_T / 60:.0f} min total")
    print(f"  New columns: odom_speed_mps + wheel_fl/fr/rl/rr_rpm")
    print(f"  Weathers: {', '.join(WEATHER_NAMES)}")
    print("="*65 + "\n")

    try:
        client = carla.Client(HOST, PORT)
        client.set_timeout(TIMEOUT_S)
        world = client.get_world()

        if world.get_map().name.split('/')[-1] != MAP_NAME:
            print(f"  Loading {MAP_NAME}...")
            world = client.load_world(MAP_NAME)
            time.sleep(8)

        orig_settings = world.get_settings()
        s = world.get_settings()
        s.synchronous_mode    = True
        s.fixed_delta_seconds = FIXED_DELTA_T
        world.apply_settings(s)
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        train_f = open(TRAIN_CSV, 'w', newline='', encoding='utf-8')
        debug_f = open(DEBUG_CSV, 'w', newline='', encoding='utf-8')
        tw = csv.DictWriter(train_f, fieldnames=TRAIN_COLS)
        dw = csv.DictWriter(debug_f, fieldnames=DEBUG_COLS)
        tw.writeheader(); dw.writeheader()

        conv      = CoordConverter()
        scheduler = SpeedScheduler(SPEED_SCHEDULE)

        for run_id in range(NUM_RUNS):
            spawn_idx    = HIGHWAY_SPAWN_INDICES[run_id % len(HIGHWAY_SPAWN_INDICES)]
            weather      = WEATHER_PRESETS[run_id]
            weather_name = WEATHER_NAMES[run_id]
            conv.ref_lat = conv.ref_lon = None
            try:
                collect_run(client, world, tm, run_id, spawn_idx,
                            weather, weather_name,
                            conv, scheduler, tw, dw, train_f, debug_f)
            except Exception as e:
                print(f"\n[WARN] Run {run_id} failed: {e} — continuing.\n")
            train_f.flush(); debug_f.flush()

        print("\n" + "="*65)
        print("  COLLECTION COMPLETE")
        print(f"  {TRAIN_CSV}")
        print(f"  Columns ({len(TRAIN_COLS)}): {TRAIN_COLS}")
        print("="*65)

    except Exception as e:
        print(f"\n[FATAL] {e}")
    finally:
        for f in [train_f, debug_f]:
            try:
                if f: f.close()
            except Exception: pass
        if orig_settings and client:
            try:
                world.apply_settings(orig_settings)
                print("  Settings restored.")
            except Exception: pass
        print("[Done]")


if __name__ == '__main__':
    main()
