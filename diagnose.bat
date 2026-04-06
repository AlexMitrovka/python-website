@echo off
chcp 65001 >nul
cd /d "%~dp0"
set LOG=%~dp0streamlit_diagnose.log
echo === Діагностика %date% %time% === > "%LOG%"

echo Перевірка Python... >> "%LOG%"
py -3 -c "import sys; print(sys.executable); print(sys.version)" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo ПОМИЛКА: py -3 не працює. Див. %LOG%
  type "%LOG%"
  pause
  exit /b 1
)

echo. >> "%LOG%"
echo pip list (streamlit)... >> "%LOG%"
py -3 -m pip show streamlit >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo import streamlit... >> "%LOG%"
py -3 -c "import streamlit as st; print('streamlit', st.__version__)" >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo import pandas, plotly, numpy... >> "%LOG%"
py -3 -c "import pandas; import plotly; import numpy; print('ok')" >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo py_compile app.py... >> "%LOG%"
py -3 -m py_compile app.py >> "%LOG%" 2>&1

echo.
echo === Результат записано в: %LOG% ===
type "%LOG%"
echo.
pause
