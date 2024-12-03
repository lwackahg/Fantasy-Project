@echo off
cd /d "%~dp0"
echo Starting Fantasy Trade Analyzer...
echo Current directory: %CD%

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not found in PATH
    pause
    exit /b 1
)

REM Check if required directories exist
if not exist "src" (
    echo Error: src directory not found
    pause
    exit /b 1
)

if not exist "src/app.py" (
    echo Error: src/app.py not found
    pause
    exit /b 1
)

REM Try to run the app
echo Running Streamlit app...
python -m streamlit run src/app.py

REM If there's an error, pause to show the message
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error running the app. See message above.
    pause
    exit /b 1
)

REM Keep window open
pause
