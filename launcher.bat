@echo off
setlocal enabledelayedexpansion
rem ===========================================================================
rem  VMAgentPro launcher - ONE entry point, ONE system.
rem
rem  Starts the GPU scheduling dashboard, the only frontend this project runs.
rem  It reads the engine's own outputs (summary.json, tasks.csv, events.csv,
rem  resource_history.csv), i.e. the ASI-trace driven reservation simulator
rem  under vmagent\gpu.
rem
rem  The original VMAgent SchedVis frontend is still in the tree at schedvis\,
rem  byte-for-byte unmodified, but it is NOT wired into this launcher any more:
rem  it only understands pickle traces of the CPU/MEM dual-NUMA environment and
rem  cannot display a single GPU result. To look at it for reference, create a
rem  Python 3.11 venv with the pinned dash 1.21.0 stack and run schedvis\run.py
rem  by hand - README_GPU.md section 3.1 carries the exact commands, taken from
rem  `git show c0bdb8b:launcher.bat` where they were last measured working.
rem
rem  Usage:
rem    launcher.bat                start the GPU dashboard and open the page
rem    launcher.bat dashboard      the same, spelled out for old shortcuts
rem
rem  NOTE: keep this file ASCII-only. cmd.exe reads .bat files in the ANSI
rem  codepage, and non-ASCII bytes here corrupt the parser mid-file. A Chinese
rem  file NAME is fine; Chinese text inside the file is not.
rem ===========================================================================
set "MODE=%~1"
if not defined MODE goto dashboard
if /i "%MODE%"=="dashboard" goto dashboard
if /i "%MODE%"=="schedvis" goto gone
if /i "%MODE%"=="both" goto gone
echo Unknown mode "%MODE%". This launcher only starts the GPU dashboard.
echo     launcher.bat              (or: launcher.bat dashboard)
exit /b 2

rem ---------------------------------------------------------------------------
rem Working directory for the server: pushd, then a plain start. Measured on
rem this machine, whose path contains non-ASCII characters, start /d "%~dp0"
rem never launches anything - the trailing backslash escapes the closing quote -
rem and neither does /d with that backslash stripped. start fails silently and
rem the run lands in the branch below, so pushd is the only form that works.
rem ---------------------------------------------------------------------------
:dashboard
set "PY="
set "PYARGS="
py -3 --version >nul 2>nul
if not errorlevel 1 (
  set "PY=py"
  set "PYARGS=-3"
)
if not defined PY set "PY=python"
if not defined PY goto nopython
pushd "%~dp0"
call :probe 8765
if not errorlevel 1 goto open
echo Starting the VMAgentPro GPU dashboard on port 8765 ...
echo A minimized window named "VMAgent Dashboard" opens; keep it open while you work.
start "VMAgent Dashboard" /min "%PY%" %PYARGS% -m vmagent.gpu.dashboard --port 8765
call :wait 8765 20
if errorlevel 1 (
  echo Failed to start the dashboard on port 8765.
  echo Check that Python 3 is installed and that port 8765 is free.
  popd
  pause
  exit /b 1
)
:open
echo http://127.0.0.1:8765/
start "" "http://127.0.0.1:8765/"
popd
exit /b 0

:gone
echo The original VMAgent SchedVis frontend is no longer started from here.
echo It reads only CPU/MEM dual-NUMA pickle traces and cannot display a GPU
echo result, so this project runs exactly one frontend: the dashboard above.
echo schedvis\ is still in the tree, unmodified, for reference.
exit /b 2

:nopython
echo No Python interpreter found. Install Python 3 and make sure either
echo     py -3
echo or
echo     python
echo works from a command prompt.
pause
exit /b 1

rem ---------------------------------------------------------------------------
rem :probe PORT -> errorlevel 0 when something already listens on PORT.
rem Both helpers read %PY% and %PYARGS% from the caller rather than taking them
rem as arguments: an empty quoted argument is still an argument, and python then
rem tries to open a file named nothing. That was a real bug, not a hypothetical
rem one, so the expansion stays bare and quoted never.
rem ---------------------------------------------------------------------------
:probe
"%PY%" %PYARGS% -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1',%~1))==0 else 1)" 2>nul
exit /b %errorlevel%

rem :wait PORT TRIES -> errorlevel 0 once the port answers, 1 after TRIES tries
:wait
for /l %%i in (1,1,%~2) do (
  "%PY%" %PYARGS% -c "import socket,sys; s=socket.socket(); s.settimeout(0.3); sys.exit(0 if s.connect_ex(('127.0.0.1',%~1))==0 else 1)" 2>nul
  if not errorlevel 1 exit /b 0
  "%SystemRoot%\System32\ping.exe" -n 2 127.0.0.1 >nul 2>nul
)
exit /b 1
