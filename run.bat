@echo off
REM Check if the virtual environment directory exists
IF NOT EXIST .venv (
    echo Creating virtual environment...
    python -m venv .venv

    REM Activate the virtual environment
    call .venv\Scripts\activate

    REM Install dependencies from requirements.txt
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Run the Python script with any passed arguments
echo Running Python script...
python app.py -a

REM