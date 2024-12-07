@echo off
REM Start the Python application inside the virtual environment

REM Activate the virtual environment
call venv\Scripts\activate

REM Run the Python script
echo Running main.py...
python main.py


pause