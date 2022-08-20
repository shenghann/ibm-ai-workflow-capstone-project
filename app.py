import time
import uvicorn
from json import JSONDecodeError
from fastapi import FastAPI, HTTPException, Request
import datalib, model, logger

description = """
How to use:

## Train Endpoint

Send a POST request to this endpoints to trigger training/re-training of the model. The `mode` parameter is optional and used only for unittests.

## Predict Endpoint

Send a POST request with your input data as a JSON payload in the following format:
```
{
            "country": "United Kingdom",
            "year": 2018,
            "month": 1,
            "day": 5
}        
```

Supported countries (top 10 revenue):
- `all` for all countries combined
- EIRE
- France
- Germany
- Hong Kong
- Netherlands
- Norway
- Portugal
- Singapore
- Spain
- United Kingdom

Returns a value for predicted revenue for the country and day.

## Logging Endpoint

View last 5 entries of logs from training and prediction pipelines. 

The `type` parameters accepts:
- `pred` for prediction logs
- `train` for training logs

"""

# init fastAPI
app = FastAPI(title='Revenue Prediction App', description=description)

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome"}

@app.post("/train")
def train(mode=None):
    start_time = time.time()
    model.process_and_train(test=(mode == 'test'))
    return {"message": f"Training completed in {(time.time() - start_time):.2}"}

@app.post('/predict')
async def predict(data : Request):
    try:
        json_data = await data.json()
        country = json_data['country']
        if not country:
            raise HTTPException(status_code=400,
                detail="Please provide a country name")
        
        y_pred = model.predict_model(country, json_data['year'], json_data['month'], json_data['day'])

        return y_pred
        
    except JSONDecodeError:
        raise HTTPException(status_code=400,
            detail="Payload is not valid JSON")

    

@app.get("/logs")
def logs(type):
    """
    API to get logs
    """
    # read log file
    log_entries = []
    if type in logger.LOG_FILES.keys():
        log_path = logger.LOG_FILES[type]
        with open(log_path, 'r') as f:
            all_lines = f.readlines()
        log_entries = all_lines[-5:]
        return log_entries
    else:
        raise HTTPException(status_code=400,
            detail='Invalid log file type')

if __name__ == "__main__":
    uvicorn.run("app:app")