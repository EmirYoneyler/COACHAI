@echo off
echo Installing dependencies using Python 3.11...
py -3.11 -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b %errorlevel%
)

echo Starting FitAI with Python 3.11...
py -3.11 -m streamlit run app.py
pause
