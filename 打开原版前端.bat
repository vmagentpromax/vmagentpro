@echo off
rem Start the ORIGINAL VMAgent SchedVis frontend (Plotly Dash 1.x, schedvis/).
rem Official entry point is schedvis/run.py, default port 8050.
rem NOTE: this file must stay ASCII-only; cmd.exe reads .bat in the ANSI codepage.
rem The pinned 2021 stack (dash==1.21.0, dash-bootstrap-components==0.13.0) does
rem not install on Python 3.12, so it needs its own Python 3.11 virtualenv.
setlocal
pushd "%~dp0schedvis"

set "PY="
if defined VMAGENT_SCHEDVIS_PY set "PY=%VMAGENT_SCHEDVIS_PY%"
if not defined PY if exist "%~dp0..\.tmp\schedvis_env\Scripts\python.exe" set "PY=%~dp0..\.tmp\schedvis_env\Scripts\python.exe"
if not defined PY if exist "%~dp0schedvis_env\Scripts\python.exe" set "PY=%~dp0schedvis_env\Scripts\python.exe"

if not defined PY goto needenv

"%PY%" -c "import dash" >nul 2>&1
if errorlevel 1 goto needdash

"%PY%" -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1',8050))==0 else 1)" >nul 2>&1
if not errorlevel 1 goto already

echo Starting original SchedVis on http://127.0.0.1:8050/ ...
echo Close the server window to stop it.
start "VMAgent SchedVis" /min "%PY%" run.py
for /l %%i in (1,1,30) do (
  "%PY%" -c "import socket,sys; s=socket.socket(); s.settimeout(0.3); sys.exit(0 if s.connect_ex(('127.0.0.1',8050))==0 else 1)" >nul 2>&1
  if not errorlevel 1 goto open
  "%SystemRoot%\System32\ping.exe" -n 2 127.0.0.1 >nul
)
echo Port 8050 did not open within 30 seconds. Check the server window above.
pause
popd
exit /b 1

:already
echo Port 8050 already has a server. Opening the page.
start "" "http://127.0.0.1:8050/"
popd
exit /b 0

:open
start "" "http://127.0.0.1:8050/"
popd
endlocal
exit /b 0

:needenv
echo SchedVis virtualenv not found. Create one with Python 3.11, NOT 3.12:
echo   C:\anaconda\python.exe -m venv "%~dp0..\.tmp\schedvis_env"
echo then install the pinned stack with this single command:
echo   "%~dp0..\.tmp\schedvis_env\Scripts\python.exe" -m pip install "numpy<2" "pandas==1.5.3" "flask==2.0.3" "werkzeug==2.0.3" "jinja2==3.0.3" "itsdangerous==2.0.1" "click==8.0.4" "flask-compress" "dash==1.21.0" "dash-bootstrap-components==0.13.0" "dash-bootstrap-templates==0.1.1" "dash-daq==0.5.0" "plotly==5.3.1"
echo Or set VMAGENT_SCHEDVIS_PY to an interpreter that already has them.
pause
exit /b 1

:needdash
echo The virtualenv exists but dash is not installed in it.
echo Use the pip install command shown when the venv is missing.
pause
exit /b 1
