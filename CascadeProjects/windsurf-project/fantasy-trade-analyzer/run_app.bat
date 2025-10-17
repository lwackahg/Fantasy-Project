@echo off
setlocal

REM --- Configuration ---
set "VENV_DIR=.venv"
set "REQUIREMENTS_FILE=requirements.txt"
set "MAIN_APP_FILE=Home.py"

REM --- Script Start ---
cd /d "%~dp0"
echo Starting Fantasy Trade Analyzer...
echo Project directory: %CD%
echo.

REM --- Python and Virtual Environment Setup ---

REM Check for python, then python3
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PYTHON_EXE=python
) else (
    where python3 >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set PYTHON_EXE=python3
    ) else (
        echo Error: Python is not found in your PATH.
        echo Please install Python 3.8+ and ensure it's added to your PATH.
        pause
        exit /b 1
    )
)
echo Found Python executable: %PYTHON_EXE%

REM Check if the virtual environment directory exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating Python virtual environment in '%VENV_DIR%'...
    %PYTHON_EXE% -m venv %VENV_DIR%
    if %ERRORLEVEL% neq 0 (
        echo Error: Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

REM Activate the virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
REM Ensure we use the venv's Python explicitly for all subsequent commands
set PYTHON_EXE="%CD%\%VENV_DIR%\Scripts\python.exe"
echo.

REM Diagnostics: show tool versions
echo Python: & %PYTHON_EXE% --version
echo Pip: & %PYTHON_EXE% -m pip --version
echo Streamlit: & %PYTHON_EXE% -m streamlit --version
echo.

REM --- Dependency Installation ---

REM Check for requirements.txt
if not exist "%REQUIREMENTS_FILE%" (
    echo Error: %REQUIREMENTS_FILE% not found. Cannot install dependencies.
    pause
    exit /b 1
)

echo Installing/updating dependencies from %REQUIREMENTS_FILE%...
%PYTHON_EXE% -m pip install -r %REQUIREMENTS_FILE%
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install dependencies. See message above.
    pause
    exit /b 1
)
echo.

REM --- Application Launch ---

REM Check for main app file
if not exist "%MAIN_APP_FILE%" (
    echo Error: Main application file not found at '%MAIN_APP_FILE%'.
    pause
    exit /b 1
)

echo Starting Streamlit application...
echo Visit the URL provided by Streamlit in your browser.
echo Press Ctrl+C in this window to stop the server.
echo.

REM Environment preferences for Streamlit
set STREAMLIT_SERVER_HEADLESS=true
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set PYTHONUTF8=1

%PYTHON_EXE% -m streamlit run "%MAIN_APP_FILE%" ^
  --server.headless true ^
  --server.address 127.0.0.1 ^
  --server.port 8501 ^
  --browser.gatherUsageStats false ^
  --logger.level=info

REM --- Script End ---
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: The application failed to start. See message above.
) else (
    echo.
    echo Application server stopped.
)

pause
endlocal