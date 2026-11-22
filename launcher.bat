@echo off
setlocal enabledelayedexpansion
rem Single launcher for both VMAgent frontends. This is the only implementation:
rem the three older launchers (the two chinese-named ones and start_dashboard.bat)
rem are one-line wrappers that call this file, so the port probing, interpreter
rem selection and error text live in exactly one place.
rem
rem NOTE: keep this file ASCII-only. cmd.exe reads .bat files in the ANSI codepage,
rem and non-ASCII bytes here corrupt the parser mid-file - a Chinese filename is
rem fine, Chinese text inside the file is not.
rem
rem Usage (the file name below is this file, spelled for the ANSI reader):
rem   launcher.bat                interactive menu
rem   launcher.bat dashboard      this repo's GPU scheduling dashboard, port 8765
rem   launcher.bat schedvis       the original VMAgent SchedVis frontend, port 8050
rem   launcher.bat both           both servers, each in its own minimized window

set "MODE=%~1"
if defined MODE goto dispatch

echo.
echo   VMAgent launcher
echo   ================
echo     [1] GPU scheduling dashboard   this repo, http://127.0.0.1:8765/
echo     [2] Original VMAgent SchedVis  schedvis,  http://127.0.0.1:8050/
echo     [3] both
echo.
set "PICK="
set /p PICK=  pick 1, 2 or 3 (Enter = 1):
if not defined PICK set "PICK=1"
if "%PICK%"=="1" goto dashboard
if "%PICK%"=="2" goto schedvis
if "%PICK%"=="3" goto both
echo Unknown choice: %PICK%
exit /b 2

:dispatch
if /i "%MODE%"=="dashboard" goto dashboard
if /i "%MODE%"=="schedvis" goto schedvis
if /i "%MODE%"=="both" goto both
echo Unknown mode "%MODE%". Expected dashboard, schedvis or both.
exit /b 2

:both
call :dashboard_bg
call :schedvis
exit /b 0

:dashboard
set "PY="
set "PYARGS="
py -3 --version >nul 2>nul
if not errorlevel 1 (
  set "PY=py"
  set "PYARGS=-3"
)
if not defined PY set "PY=python"
pushd "%~dp0"
call :probe 8765
if not errorlevel 1 goto dashboard_open
echo Starting the VMAgent GPU dashboard on port 8765 ...
echo A minimized window named "VMAgent Dashboard" opens; keep it open while you work.
start "VMAgent Dashboard" /min "%PY%" %PYARGS% -m vmagent.gpu.dashboard --port 8765
call :wait 8765 20
if errorlevel 1 (
  echo Failed to start the VMAgent dashboard on port 8765.
  echo Check that Python 3 is installed and that port 8765 is free.
  popd
  pause
  exit /b 1
)
:dashboard_open
echo http://127.0.0.1:8765/
start "" "http://127.0.0.1:8765/"
popd
exit /b 0

:dashboard_bg
set "PY="
set "PYARGS="
py -3 --version >nul 2>nul
if not errorlevel 1 (
  set "PY=py"
  set "PYARGS=-3"
)
if not defined PY set "PY=python"
pushd "%~dp0"
call :probe 8765
if not errorlevel 1 goto dashboard_bg_open
start "VMAgent Dashboard" /min "%PY%" %PYARGS% -m vmagent.gpu.dashboard --port 8765
:dashboard_bg_open
start "" "http://127.0.0.1:8765/"
popd
exit /b 0

:schedvis
rem The pinned 2021 stack (dash 1.21.0, dbc 0.13.0) does not install on Python 3.12,
rem so SchedVis needs its own Python 3.11 virtualenv. Search order: explicit override,
rem the shared .tmp location, then a venv next to schedvis.
set "PY="
if defined VMAGENT_SCHEDVIS_PY set "PY=%VMAGENT_SCHEDVIS_PY%"
if not defined PY if exist "%~dp0..\.tmp\schedvis_env\Scripts\python.exe" set "PY=%~dp0..\.tmp\schedvis_env\Scripts\python.exe"
if not defined PY if exist "%~dp0schedvis_env\Scripts\python.exe" set "PY=%~dp0schedvis_env\Scripts\python.exe"
if not defined PY goto needenv

"%PY%" -c "import dash" >nul 2>&1
if errorlevel 1 goto needdash

set "PYARGS="
pushd "%~dp0schedvis"
call :probe 8050
if not errorlevel 1 goto schedvis_open
echo Starting the original VMAgent SchedVis on http://127.0.0.1:8050/ ...
echo Close the "VMAgent SchedVis" window to stop the server.
start "VMAgent SchedVis" /min "%PY%" run.py
call :wait 8050 30
if errorlevel 1 (
  echo Port 8050 did not open within 30 seconds. Check the server window.
  popd
  pause
  exit /b 1
)
:schedvis_open
echo http://127.0.0.1:8050/
start "" "http://127.0.0.1:8050/"
popd
exit /b 0

rem ---- shared helpers -------------------------------------------------------
rem Both read %PY% and %PYARGS% from the caller instead of taking them as
rem arguments. Passing them positionally looked tidy until schedvis needed "no extra
rem interpreter arguments": an empty quoted argument is still an argument, so python
rem then tries to run a file named nothing. %PYARGS% expands to nothing when unset,
rem which is what the command line actually needs.
rem :probe PORT -> errorlevel 0 when something is already listening
:probe
"%PY%" %PYARGS% -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1',%~1))==0 else 1)" >nul 2>&1
exit /b %errorlevel%

rem :wait PORT TRIES -> errorlevel 0 once the port answers, 1 after TRIES
:wait
for /l %%i in (1,1,%~2) do (
  "%PY%" %PYARGS% -c "import socket,sys; s=socket.socket(); s.settimeout(0.3); sys.exit(0 if s.connect_ex(('127.0.0.1',%~1))==0 else 1)" >nul 2>&1
  if not errorlevel 1 exit /b 0
  "%SystemRoot%\System32\ping.exe" -n 2 127.0.0.1 >nul
)
exit /b 1

:needenv
echo SchedVis virtualenv not found. Create one with Python 3.11, NOT 3.12:
echo   py -3.11 -m venv "%~dp0..\.tmp\schedvis_env"
echo then install the pinned stack with this single command:
echo   "%~dp0..\.tmp\schedvis_env\Scripts\python.exe" -m pip install "numpy<2" "pandas==1.5.3" "flask==2.0.3" "werkzeug==2.0.3" "jinja2==3.0.3" "itsdangerous==2.0.1" "click==8.0.4" "flask-compress" "dash==1.21.0" "dash-bootstrap-components==0.13.0" "dash-bootstrap-templates==0.1.1" "dash-daq==0.5.0" "plotly==5.3.1"
echo Or set VMAGENT_SCHEDVIS_PY to an interpreter that already has them.
echo The GPU dashboard (option 1) does not need any of this; it runs on stdlib Python.
pause
exit /b 1

:needdash
echo The SchedVis virtualenv exists but dash is not installed in it.
echo Use the pip install command shown when the venv is missing.
pause
exit /b 1
