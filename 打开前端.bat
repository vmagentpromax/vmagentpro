@echo off
setlocal
pushd "%~dp0"
set "PY=python"
set "PYARGS="
py -3 --version >nul 2>nul
if not errorlevel 1 (
  set "PY=py"
  set "PYARGS=-3"
)
"%PY%" %PYARGS% -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1',8765))==0 else 1)" >nul 2>&1
if errorlevel 1 (
  start "VMAgent Dashboard" /min "%PY%" %PYARGS% -m vmagent.gpu.dashboard --port 8765
  for /l %%i in (1,1,20) do (
    "%PY%" %PYARGS% -c "import socket,sys; s=socket.socket(); s.settimeout(0.3); sys.exit(0 if s.connect_ex(('127.0.0.1',8765))==0 else 1)" >nul 2>&1
    if not errorlevel 1 goto open
    "%SystemRoot%\System32\ping.exe" -n 2 127.0.0.1 >nul
  )
  echo Failed to start the VMAgent dashboard on port 8765.
  echo Check that Python 3 is installed and port 8765 is free.
  pause
  exit /b 1
)
:open
start "" "http://127.0.0.1:8765/"
popd
endlocal
