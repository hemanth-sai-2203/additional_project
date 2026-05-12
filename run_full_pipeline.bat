@echo off
call carla_env37\Scripts\activate.bat

echo ============================================================
echo  STEP 1: Retrain Physics LSTM on new data (48k rows, 6 runs)
echo ============================================================
python -u lstm\train_lstm_physics.py

echo.
echo ============================================================
echo  STEP 2: Generate Physics+Odometry EKF Baseline (v2)
echo ============================================================
python -u generate_physics_baseline.py

echo.
echo ============================================================
echo  STEP 3: Train LSTM Locator on Physics Baseline v2
echo ============================================================
python -u lstm\train_lstm_locator.py


