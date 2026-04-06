@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   АКБ: аналіз розряду з CSV — Streamlit
echo   %~dp0app.py
echo ========================================
echo.

where py >nul 2>&1
if errorlevel 1 (
    echo ПОМИЛКА: команда "py" не знайдена. Встановіть Python з python.org
    pause
    exit /b 1
)

echo Python:
py -3 -c "import sys; print(sys.executable); print(sys.version)"
if errorlevel 1 ( pause & exit /b 1 )

echo.
py -3 -c "import streamlit as st; print('Streamlit', st.__version__)" 2>nul
if errorlevel 1 (
    echo Встановлення залежностей...
    py -3 -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 ( pause & exit /b 1 )
)

echo.
echo Браузер: http://localhost:8501
echo Зупинка: у цьому вікні натисніть Ctrl+C
echo.
REM headless — не відкривати браузер автоматично (інколи дає збій)
py -3 -m streamlit run "%~dp0app.py" --server.headless true

echo.
echo Код виходу: %errorlevel%
if %errorlevel% neq 0 (
    echo.
    echo Код -1 часто буває на Python 3.14. Спробуйте Python 3.12:
    echo   https://www.python.org/downloads/
    echo Потім: py -3.12 -m pip install -r requirements.txt
    echo        py -3.12 -m streamlit run app.py
    echo.
    echo Діагностика: запустіть diagnose.bat
)
pause
