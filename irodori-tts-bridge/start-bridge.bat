@echo off
REM ============================================================
REM  Irodori-TTS bridge launcher for FlowMinds://AI
REM  Place this file and bridge.py in the Irodori-TTS folder
REM  (the folder that contains "irodori_tts" and "pyproject.toml").
REM
REM  Endpoints:
REM    POST http://127.0.0.1:8080/api/irodori/v1/tts
REM    POST http://127.0.0.1:8080/api/voicedesign/v1/tts
REM ============================================================

REM --- PyTorch build -------------------------------------------
REM   cu128 : NVIDIA GPU (recommended)
REM   cpu   : no NVIDIA GPU (works, but very slow)
set IRODORI_EXTRA=cu128

REM --- Model (optional) ----------------------------------------
REM   Default: Aratako/Irodori-TTS-v4-Small
REM   To use v3 instead, remove "REM " from the next two lines.
REM set IRODORI_CHECKPOINT_IRODORI=Aratako/Irodori-TTS-500M-v3
REM set IRODORI_CHECKPOINT_VOICEDESIGN=Aratako/Irodori-TTS-600M-v3-VoiceDesign

set PATH=%USERPROFILE%\.local\bin;%PATH%
REM huggingface-hub fails with WinError 1314 when symlinks are not permitted.
set HF_HUB_DISABLE_SYMLINKS=1
cd /d "%~dp0"

if not exist "irodori_tts\" (
  echo [start-bridge] ERROR: "irodori_tts" folder not found.
  echo [start-bridge] Put start-bridge.bat and bridge.py inside the Irodori-TTS folder.
  pause
  exit /b 1
)
if not exist "bridge.py" (
  echo [start-bridge] ERROR: bridge.py not found next to this file.
  pause
  exit /b 1
)
where uv >nul 2>nul
if errorlevel 1 (
  echo [start-bridge] ERROR: uv is not installed.
  echo [start-bridge] Install it: https://docs.astral.sh/uv/getting-started/installation/
  pause
  exit /b 1
)

echo [start-bridge] Starting Irodori-TTS bridge on http://127.0.0.1:8080 ...
echo [start-bridge] PyTorch build: %IRODORI_EXTRA%
echo [start-bridge] First run installs packages and downloads the model (several GB). Please wait.
REM The --extra flag is required every time: without it "uv run" re-syncs PyTorch back to the CPU build.
REM fastapi / uvicorn / hf_xet are bridge-only dependencies, added with --with (pyproject.toml stays untouched).
uv run --extra %IRODORI_EXTRA% --with fastapi --with "uvicorn[standard]" --with hf_xet python -m uvicorn bridge:app --host 127.0.0.1 --port 8080
if errorlevel 1 pause
