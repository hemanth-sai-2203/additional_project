@echo off

REM Set CARLA egg path
set PYTHONPATH=<CARLA_PATH>\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg;%PYTHONPATH%

echo ============================================================
echo  SOTA CARLA LIVE DEMO (Autopilot Mode)
echo ============================================================
echo  Make sure CarlaUE4.exe is ALREADY RUNNING before this!
echo.

REM Activate environment
call carla_env37\Scripts\activate.bat
python sota_carla_demo.py

pause

