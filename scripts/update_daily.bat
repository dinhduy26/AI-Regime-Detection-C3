@echo off
REM ==================================================================
REM  Daily data update (Windows). Run after 07:05 Vietnam time,
REM  when the previous UTC daily candle has closed.
REM  Download latest closed candles -> clean -> features -> tables/figures
REM  With split.mode = fixed_dates: train/validation/scaler do NOT change;
REM  new days are only appended to the end of the test set.
REM  Log: outputs\metadata\update_log.txt
REM ==================================================================
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
echo ===== %date% %time% ===== >> outputs\metadata\update_log.txt
python run_pipeline.py --download >> outputs\metadata\update_log.txt 2>&1
python -m pytest -q >> outputs\metadata\update_log.txt 2>&1
