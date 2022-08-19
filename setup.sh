#!/bin/bash

# Create environment
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
deactivate
echo Install complete