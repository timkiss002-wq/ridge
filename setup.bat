@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/2] Tao moi truong ao .venv ...
python -m venv .venv
call .venv\Scripts\activate.bat
echo [2/2] Cai thu vien ...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Cai dat xong. Nhay dup run_app.bat de mo chuong trinh.
pause
