@echo off
cd /d "%~dp0"
python -m streamlit run model/app.py
pause