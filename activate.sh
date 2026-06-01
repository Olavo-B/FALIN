#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Display information
echo -e "\e[32m[SUCCESS]\e[0m Virtual environment activated for project: FALIN"
echo -e "\e[34m[INFO]\e[0m Python version: $(python --version)"
echo -e "\e[34m[INFO]\e[0m Pip version: $(pip --version)"
echo -e "\e[33m[INFO]\e[0m To install dependencies, run: pip install -r requirements.txt"
echo -e "\e[33m[INFO]\e[0m To deactivate, run: deactivate"

# Execute bash to keep the session active
exec bash
