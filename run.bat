@echo off
echo.
echo ========================================
echo   Student Analytics Dashboard
echo   Starting Streamlit App...
echo ========================================
echo.

cd /d "%~dp0"
"C:/Program Files/Python313/python.exe" -m streamlit run app.py --server.headless=true

pause
