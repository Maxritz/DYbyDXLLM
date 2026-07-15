@echo off
rem -----------------------------------------------------------------
rem  Batch script to benchmark the three models with DybyDx CLI.
rem  It runs each model, logs the full console output, and writes the
rem  generated 200‑word essay to a separate .txt file.
rem -----------------------------------------------------------------

:: -----------------------------------------------------------------
::  Paths – models are in C:\Users\rina0423\Desktop\DX\
:: -----------------------------------------------------------------
set "DYBYDX_EXE=%~dp0build\bin\Release\DybyDx.exe"
set "MODEL_DIR=C:\Users\rina0423\Desktop\DX"
set "MINICPM_MODEL=%MODEL_DIR%\MiniCPM-V-4_6-Q8_0.gguf"
set "GEMMA_MODEL=%MODEL_DIR%\gemma-4-E4B-it-Q4_K_M.gguf"
set "PHI3_MODEL=%MODEL_DIR%\Phi-3-mini-4k-instruct-q4.gguf"

:: Prompt (same for all three runs)
set "PROMPT=Write a 200-word essay about the importance of sustainable computing."

:: Output log file (all console output goes here)
set "LOG_FILE=%~dp0benchmark_log.txt"
if exist "%LOG_FILE%" del "%LOG_FILE%"

echo ============================= >> "%LOG_FILE%"
echo Benchmark run started at %date% %time% >> "%LOG_FILE%"
echo ============================= >> "%LOG_FILE%"

rem -----------------------------------------------------------------
rem  1) MiniCPM‑V‑4_6‑Q8_0
rem -----------------------------------------------------------------
echo ----- Running MiniCPM‑V‑4_6‑Q8_0 ----- >> "%LOG_FILE%"
if not exist "%MINICPM_MODEL%" (echo [ERROR] MiniCPM model not found at %MINICPM_MODEL% && exit /b 1)
"%DYBYDX_EXE%" ^
  --model "%MINICPM_MODEL%" ^
  --prompt "%PROMPT%" ^
  --n-predict 200 ^
  --output "miniCPM_essay.txt" ^
  >>"%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (echo [ABORTED] MiniCPM failed - model not loaded >> "%LOG_FILE%" && exit /b 1)

rem -----------------------------------------------------------------
rem  2) Gemma‑4‑E4B‑it‑Q4_K_M
rem -----------------------------------------------------------------
echo. >> "%LOG_FILE%"
echo ----- Running Gemma‑4‑E4B‑it‑Q4_K_M ----- >> "%LOG_FILE%"
if not exist "%GEMMA_MODEL%" (echo [ERROR] Gemma model not found at %GEMMA_MODEL% && exit /b 1)
"%DYBYDX_EXE%" ^
  --model "%GEMMA_MODEL%" ^
  --prompt "%PROMPT%" ^
  --n-predict 200 ^
  --output "gemma_essay.txt" ^
  >>"%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (echo [ABORTED] Gemma failed - model not loaded >> "%LOG_FILE%" && exit /b 1)

rem -----------------------------------------------------------------
rem  3) Phi‑3‑mini‑4k‑instruct‑q4
rem -----------------------------------------------------------------
echo. >> "%LOG_FILE%"
echo ----- Running Phi‑3‑mini‑4k‑instruct‑q4 ----- >> "%LOG_FILE%"
if not exist "%PHI3_MODEL%" (echo [ERROR] Phi-3 model not found at %PHI3_MODEL% && exit /b 1)
"%DYBYDX_EXE%" ^
  --model "%PHI3_MODEL%" ^
  --prompt "%PROMPT%" ^
  --max-tokens 200 ^
  --output "phi3_essay.txt" ^
  >>"%LOG_FILE%" 2>&1

rem -----------------------------------------------------------------
rem  Finished
rem -----------------------------------------------------------------
echo. >> "%LOG_FILE%"
echo ============================= >> "%LOG_FILE%"
echo Benchmark run finished at %date% %time% >> "%LOG_FILE%"
echo ============================= >> "%LOG_FILE%"

echo.
echo ==========================================================
echo Benchmark completed.
echo Results are in:
echo   - %LOG_FILE%          (full console log, architecture & throughput)
echo   - miniCPM_essay.txt   (MiniCPM‑V output)
echo   - gemma_essay.txt     (Gemma‑4 output)
echo   - phi3_essay.txt      (Phi‑3 output)
echo ==========================================================
rem pause