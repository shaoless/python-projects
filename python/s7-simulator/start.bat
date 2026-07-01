@echo off
title S7-1200 PLC Simulators
cd /d "%~dp0"

echo.
echo ================================================
echo   S7-1200 PLC Simulators
echo   启动 3 个实例: 端口 102, 1103, 1104
echo ================================================
echo.
echo 安装依赖...
pip install -q python-snap7 2>nul
echo.

echo 启动 端口 102 ...
start "S7-Sim-102" cmd /k "cd /d %~dp0 && python server.py --port 102"

echo 启动 端口 1103 ...
start "S7-Sim-1103" cmd /k "cd /d %~dp0 && python server.py --port 1103"

echo 启动 端口 1104 ...
start "S7-Sim-1104" cmd /k "cd /d %~dp0 && python server.py --port 1104"

echo.
echo 所有模拟器已启动!
echo   127.0.0.1:102  (默认)
echo   127.0.0.1:1103
echo   127.0.0.1:1104
echo.
pause
