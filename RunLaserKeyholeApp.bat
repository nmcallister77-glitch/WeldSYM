@echo off
setlocal
cd /d "%~dp0"

:: Console-visible debug launcher (use RunLaserKeyholeApp.vbs for a silent launch)
pythonw "keyhole-cfd\app\laserkeyhole_app.pyw"
if %errorlevel% neq 0 (
    echo pythonw failed, falling back to python ...
    python "keyhole-cfd\app\laserkeyhole_app.pyw"
    pause
)
