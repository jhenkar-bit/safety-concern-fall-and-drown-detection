@echo off
rem Starts the always-on SafeGuard service and restarts it if it ever stops.
rem EDIT THE NEXT LINE if your .venv folder is somewhere else.
set VENV=C:\Users\Lenovo\Downloads\files\safeguard-ai\.venv

cd /d "%~dp0"
if not exist "%VENV%\Scripts\activate.bat" (
  echo Cannot find the virtual environment at %VENV%
  echo Open this file in Notepad and fix the "set VENV=" line.
  pause
  exit /b 1
)
call "%VENV%\Scripts\activate.bat"
:loop
echo [%date% %time%] Starting SafeGuard on http://localhost:8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
echo [%date% %time%] SafeGuard stopped. Restarting in 10 seconds. Close this window to quit.
timeout /t 10 /nobreak >nul
goto loop
