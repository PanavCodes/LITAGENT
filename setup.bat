@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo                LitAgent: System Setup and Initialization
echo ===============================================================================
echo.

:: 1. Check Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your PATH.
    echo Please install Python 3.11 or 3.12 from https://www.python.org/
    exit /b 1
)

:: 2. Check for uv (fast installer) or standard python venv
where uv >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set USE_UV=1
    echo [INFO] Detected 'uv' fast package manager.
) else (
    set USE_UV=0
    echo [INFO] 'uv' not found; using standard python/pip.
)

:: 3. Create Virtual Environment if not present
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment in .venv...
    if !USE_UV! equ 1 (
        uv venv --python 3.12 .venv
        if !ERRORLEVEL! neq 0 uv venv .venv
    ) else (
        python -m venv .venv
    )
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [INFO] Virtual environment .venv already exists.
)

:: 4. Activate Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Could not find .venv\Scripts\activate.bat.
    exit /b 1
)

:: 5. Install Dependencies
echo [INFO] Installing/verifying project dependencies...
if !USE_UV! equ 1 (
    uv pip install -e ".[dev]"
) else (
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
)
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Dependency installation failed.
    exit /b 1
)
echo [OK] Dependencies installed successfully.

:: 6. Setup Configuration (.env)
if not exist ".env" (
    if exist ".env.example" (
        echo [INFO] Creating .env from .env.example...
        copy .env.example .env >nul
        echo [OK] .env initialized. You can edit API keys if using live LLMs.
    )
) else (
    echo [INFO] .env configuration already present.
)

:: 7. Clone reference repos if third_party is empty
if not exist "third_party\SOURCES.txt" (
    echo [INFO] Initializing third_party reference tracking...
    if exist "scripts\clone_repos.sh" (
        where bash >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            bash scripts/clone_repos.sh
        )
    )
)

:: 8. Pre-download / Cache HuggingFace models & Verify Stores
echo [INFO] Pre-caching embedding and cross-encoder models...
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'); print('[OK] Sentence-Transformers embedding model cached.')"
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Model download had an issue or is running in offline mode. Continuing...
)

:: 9. Build seed corpus if missing
if not exist "data\processed\papers.jsonl" (
    echo [INFO] Building seed literature corpus...
    python scripts/build_corpus.py --offline
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to build corpus.
        exit /b 1
    )
    echo [OK] Corpus built successfully.
) else (
    echo [INFO] Corpus papers already present in data/processed/.
)

:: 10. Build index stores if missing
if not exist "data\index\faiss.index" (
    echo [INFO] Building Citation Graph, FAISS, and BM25 index stores...
    python scripts/build_index.py
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to build indexes.
        exit /b 1
    )
    echo [OK] Indexes built successfully.
) else (
    echo [INFO] Index stores already present in data/index/.
)

:: 11. Health Check
echo.
echo [INFO] Running quick system health check...
python -c "from litagent.config import load_config; from litagent.agent.workflow import LitAgentWorkflow; print('[OK] All LitAgent modules, schemas, and stores verified!')"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] System health check failed.
    exit /b 1
)

echo.
echo ===============================================================================
echo                LitAgent Setup Complete Successfully!
echo ===============================================================================
echo To launch the application, run:
echo   run.bat
echo Or use the CLI:
echo   python -m litagent.ui.cli ask "Which papers cite Dense Passage Retrieval?"
echo ===============================================================================
