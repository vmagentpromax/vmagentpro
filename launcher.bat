@echo off
setlocal enabledelayedexpansion
rem Launcher for this repository's GPU scheduling dashboard. This is the only
rem implementation: the older launchers (two chinese-named ones and
rem start_dashboard.bat) are one-line wrappers that call this file, so the port
rem probing, interpreter selection and error text live in exactly one place.
rem
rem The original VMAgent frontends (SchedVis dashboard, schedgym CPU/NUMA
rem environment, RL learners) were removed from this working tree; see
rem README_GPU.md section 9 for what was worth borrowing and how to get the
rem original files back out of git history.
rem
rem NOTE: keep this file ASCII-only. cmd.exe reads .bat files in the ANSI
rem codepage, and non-ASCII bytes here corrupt the parser mid-file - a Chinese
rem filename is fine, Chinese text inside the file is not.
rem
rem Usage (the file name below is this file, spelled for the ANSI reader):
rem   launcher.bat                start the GPU dashboard and open the page
rem   launcher.bat dashboard      same thing, spelled out, for the wrappers

set "PY="
set "PYARGS="
py -3 --version >nul 2>nul
if not errorlevel 1 (
  set "PY=py"
  set "PYARGS=-3"
)
if not defined PY set "PY=python"
pushd "%~dp0"

echo   VMAgent GPU scheduling dashboard
echo   ==================================
call :probe 8765
if not errorlevel 1 (
  echo Port 8765 is already serving; opening the page.
  goto open
)

echo Starting on http://127.0.0.1:8765/ ...
echo A minimized window named "VMAgent Dashboard" opens; keep it open while you work.
start "VMAgent Dashboard" /min "%PY%" %PYARGS% -m vmagent.gpu.dashboard --port 8765
call :wait 8765 20
if errorlevel 1 (
  echo Failed to start the dashboard on port 8765.
  echo Check that Python 3 is installed and that the port is free.
  popd
  pause
  exit /b 1
)

:open
echo http://127.0.0.1:8765/
start "" "http://127.0.0.1:8765/"
popd
exit /b 0

rem ---- shared helpers -------------------------------------------------------
rem Both read %PY% and %PYARGS% from the caller instead of taking them as
rem arguments: an empty quoted argument is still an argument, so python then
rem tries to run a file named nothing. %PYARGS% expands to nothing when unset,
rem which is what the command line actually needs.
rem :probe PORT -> errorlevel 0 when something is already listening
:probe
"%PY%" %PYARGS% -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1',%~1))==0 else 1)" >nul 2>nul
exit /b %errorlevel%

rem :wait PORT TRIES -> errorlevel 0 once the port answers, 1 after TRIES
:wait
for /l %%i in (1,1,%~2) do (
  "%PY%" %PYARGS% -c "import socket,sys; s=socket.socket(); s.settimeout(0.3); sys.exit(0 if s.connect_ex(('127.0.0.1',%~1))==0 else 1)" >nul 2>nul
  if not errorlevel 1 exit /b 0
  "%SystemRoot%\System32\ping.exe" -n 2 127.0.0.1 >nul
)
exit /b 1
