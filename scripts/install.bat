@echo off
rem MaxGaffer one-double-click installer for 3ds Max 2026.
rem No required pip packages (stdlib floor) — Pillow is installed as an OPTIONAL upgrade
rem (better JPEG reference ingestion + slimmer LLM payloads); failure there is non-fatal.
setlocal EnableDelayedExpansion

set "REPO=%~dp0.."
for %%I in ("%REPO%") do set "REPO=%%~fI"

set "MAXPY=C:\Program Files\Autodesk\3ds Max 2026\Python\python.exe"
if not exist "%MAXPY%" (
  echo [!] Could not find Max 2026 Python at:
  echo     "%MAXPY%"
  echo     Edit the MAXPY line in this script, then re-run.
  pause & exit /b 1
)

echo(
echo === MaxGaffer install ===
echo repo: %REPO%
echo(

echo [1/3] optional: installing Pillow into Max's Python user-site...
"%MAXPY%" -m ensurepip --upgrade >nul 2>&1
"%MAXPY%" -m pip install --target "%APPDATA%\Python\Python311\site-packages" pillow
if errorlevel 1 echo     (Pillow install failed — fine, MaxGaffer runs without it)

echo [2/3] registering the startup macro...
rem language folder is ENU only on English installs — copy into every initialized
rem language profile, fall back to creating ENU when none exist yet
set "MAXUSER=%LOCALAPPDATA%\Autodesk\3dsMax\2026 - 64bit"
set "COPIED=0"
for /d %%L in ("%MAXUSER%\*") do (
  if exist "%%L\scripts" (
    if not exist "%%L\scripts\startup" mkdir "%%L\scripts\startup"
    copy /Y "%REPO%\maxgaffer\startup\maxgaffer_startup.py" "%%L\scripts\startup\" >nul && set "COPIED=1"
  )
)
if "!COPIED!"=="0" (
  set "STARTUP=%MAXUSER%\ENU\scripts\startup"
  if not exist "!STARTUP!" mkdir "!STARTUP!"
  copy /Y "%REPO%\maxgaffer\startup\maxgaffer_startup.py" "!STARTUP!\" >nul
  if errorlevel 1 ( echo [!] could not copy the startup script & pause & exit /b 1 )
)

echo [3/3] recording the clone path...
"%MAXPY%" -c "import json,os,sys;d=os.path.join(os.environ['LOCALAPPDATA'],'MaxGaffer');os.makedirs(d,exist_ok=True);p=os.path.join(d,'config.json');c=(json.load(open(p)) if os.path.exists(p) else {});c['repo_path']=sys.argv[1];json.dump(c,open(p,'w'),indent=1)" "%REPO%"
setx MAXGAFFER "%REPO%" >nul

echo(
echo === done ===
echo Restart 3ds Max, then Customize ^> Customize User Interface ^> category MaxGaffer ^>
echo drag the action onto a toolbar. Click it. If MaxDirector is installed its oc_ key is
echo borrowed automatically; otherwise paste yours in Settings and Test gateway.
echo(
pause
