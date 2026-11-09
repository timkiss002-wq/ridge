@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
echo Dang mo chuong trinh tai http://localhost:8501 ...
python -m streamlit run app.py
pause
