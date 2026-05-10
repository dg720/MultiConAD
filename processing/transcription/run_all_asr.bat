@echo off
cd /d "%~dp0"

echo ============================================================
echo  MultiConAD ASR Pipeline - Whisper Large-v3
echo ============================================================

echo.
echo [1/6] ds7 (59 files - Greek tasks)
python ASR_audio_dataset.py --dataset ds7
if errorlevel 1 goto error

echo.
echo [2/6] ds5 (95 files - Greek tasks)
python ASR_audio_dataset.py --dataset ds5
if errorlevel 1 goto error

echo.
echo [3/6] ds3 (250 files - Greek recursive)
python ASR_audio_dataset.py --dataset ds3
if errorlevel 1 goto error

echo.
echo [4/6] taukadial_test (120 files - EN/ZH)
python ASR_audio_dataset.py --dataset taukadial_test
if errorlevel 1 goto error

echo.
echo [5/6] taukadial_train (387 files - EN/ZH)
python ASR_audio_dataset.py --dataset taukadial_train
if errorlevel 1 goto error

echo.
echo ============================================================
echo  All done. Check data\processed\transcriptions\ for output JSON files.
echo ============================================================
goto end

:error
echo.
echo ERROR: last dataset failed - check output above.
exit /b 1

:end
