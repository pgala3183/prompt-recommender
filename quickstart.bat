@echo off
REM Quick start script for Windows

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Generating synthetic data (500 runs)...
python scripts\generate_synthetic_data.py --num-runs 500

echo.
echo ============================================
echo Setup complete!
echo.
echo To start the API server:
echo   python -m uvicorn src.api.main:app --reload
echo.
echo To start the dashboard:
echo   streamlit run src\dashboard\app.py
echo.
echo ============================================
