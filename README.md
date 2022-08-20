# IBM AI Enterprise Workflow Capstone Project
Submission for the IBM AI Enterprise Workflow Capstone project. 

## Overview
This repository contains all files and components required by the captstone project requirements, including
- Data ingestion and model scripts
- API
- Unit Tests
- Docker image for the project
- Jupyter notebook for report findings

## Running the Project
### Using Docker
Build docker image
```
docker build -t predict-revenue-app .
```
Check if image exists
```
docker image ls
```
Run the container
```
docker run -p 8000:8000 predict-revenue-app
```
### Local Run
Setup local python environment by running provided setup script, which installs a virtual environment in `env` folder and all required packages
```
./setup.sh
```
Activate virtual environment
```
source env/bin/activate
```
Start API app
```
python app.py
```
**Recommended**: Open http://localhost:8000/docs in your browser to the see API documentations to start. Endpoints are provided for:
- Training
- Prediction
- Viewing logs

## Running tests

To test the app
```
python run_tests.py
```

## Performance Monitoring

My findings on the production data and how the trained model fared is detailed in last section of `Report Notebook.ipynb`