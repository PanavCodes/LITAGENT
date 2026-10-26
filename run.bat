@echo off
setlocal enabledelayedexpansion

:: Check if .venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run setup.bat first!
    exit /b 1
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Handle CLI flags if passed
if "%1"=="ui" goto launch_ui
if "%1"=="app" goto launch_ui
if "%1"=="ask" goto launch_ask
if "%1"=="eval" goto launch_eval
if "%1"=="test" goto launch_test
if "%1"=="trace" goto launch_trace
if "%1"=="index" goto launch_index
if not "%1"=="" goto usage

:: Interactive Menu
:menu
cls
echo ===============================================================================
echo                LitAgent: Agentic Hybrid RAG Execution Runner
echo ===============================================================================
echo.
echo Please select an option to run:
echo.
echo   [1] Launch Gradio Web UI (Interactive Chat + Execution Inspector) [DEFAULT]
echo   [2] Ask a Research Question via Command Line (CLI)
echo   [3] Run Benchmark Evaluation Suite (Ablations, RAGAS, Calibration)
echo   [4] Run Complete Unit Test Suite (Offline with Fakes)
echo   [5] View Recent Execution Traces & Latency Breakdown
echo   [6] Rebuild Search Indexes (Graph + FAISS Dense + BM25 Sparse)
echo   [7] Exit
echo.
set /p choice="Enter choice [1-7, default: 1]: "

if "%choice%"=="" set choice=1
if "%choice%"=="1" goto launch_ui
if "%choice%"=="2" goto prompt_ask
if "%choice%"=="3" goto launch_eval
if "%choice%"=="4" goto launch_test
if "%choice%"=="5" goto launch_trace
if "%choice%"=="6" goto launch_index
if "%choice%"=="7" goto end

echo [ERROR] Invalid choice. Please select 1 through 7.
pause
goto menu

:launch_ui
echo.
echo [INFO] Launching LitAgent Gradio Web Application...
echo Point your browser to: http://127.0.0.1:7860
python -m litagent.ui.app_gradio
goto end

:prompt_ask
echo.
set /p q="Enter research question: "
if "!q!"=="" set q="Which papers in the corpus cite Dense Passage Retrieval?"
python -m litagent.ui.cli ask "!q!"
pause
goto menu

:launch_ask
shift
set "FULL_QUERY="
:loop_args
if "%~1"=="" goto run_ask
set "FULL_QUERY=!FULL_QUERY! %~1"
shift
goto loop_args
:run_ask
if "!FULL_QUERY!"=="" set "FULL_QUERY=Which papers in the corpus cite Dense Passage Retrieval?"
python -m litagent.ui.cli ask "!FULL_QUERY!"
goto end

:launch_eval
echo.
echo [INFO] Running complete 5-way ablation benchmarks and uncertainty calibration...
python -m litagent.eval.report
echo [OK] Benchmark complete. Results written to results/
pause
goto menu

:launch_test
echo.
echo [INFO] Running all unit tests across Phases 0 through 7...
pytest tests/unit/
pause
goto menu

:launch_trace
echo.
python scripts/view_trace.py
pause
goto menu

:launch_index
echo.
echo [INFO] Rebuilding Citation Graph, FAISS, and BM25 index stores...
python scripts/build_index.py
pause
goto menu

:usage
echo.
echo LitAgent Runner Usage:
echo   run.bat              - Opens interactive menu
echo   run.bat ui           - Launches Gradio Web UI
echo   run.bat ask "query"  - Asks a question via CLI
echo   run.bat eval         - Runs ablation and calibration benchmark
echo   run.bat test         - Runs full test suite
echo   run.bat trace        - Views recent execution traces
echo   run.bat index        - Rebuilds corpus indexes
echo.
goto end

:end
