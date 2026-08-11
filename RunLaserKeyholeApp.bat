@echo off
setlocal

cd /d "%~dp0"

pythonw "keyhole-cfd\app\laserkeyhole_app.pyw"
if %errorlevel% neq 0 (
    echo pythonw not found or failed, trying python ...
    python "keyhole-cfd\app\laserkeyhole_app.pyw"
)

pause
