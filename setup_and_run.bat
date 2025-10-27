@echo off
echo ================================================
echo  Student Disengagement Prediction System
echo  Quick Setup and Launch
echo ================================================
echo.

echo [1/4] Installing dependencies...
pip install -r requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo  DONE - Dependencies installed

echo.
echo [2/4] Generating student data...
python generate_data.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to generate data
    pause
    exit /b 1
)

echo.
echo [3/4] Preprocessing data...
python preprocess.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to preprocess
    pause
    exit /b 1
)

echo.
echo [4/4] Training ML model...
python train_model.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to train model
    pause
    exit /b 1
)

echo.
echo ================================================
echo  Setup Complete! 
echo ================================================
echo.
echo Launching dashboard...
echo Open your browser to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py
