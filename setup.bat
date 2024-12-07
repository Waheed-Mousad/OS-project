@echo off
REM Set up Python environment and install requirements

echo Setting up Python environment...

REM Create a virtual environment named "env"
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install required packages from requirements.txt
if exist requirements.txt (
    echo Installing packages from requirements.txt...
    pip install -r requirements.txt
) else (
    echo requirements.txt not found. Skipping package installation.
)

echo Environment setup complete!
pause
