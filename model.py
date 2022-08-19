import time
import joblib
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import datalib, logger

MODEL_ROOT = Path('models')
MODEL_VERSION = 1.0
MODEL_VERSION_NOTE = 'first rev model'

def process_and_train(test=False):
    """
    Load data and train model
    """
    MODEL_ROOT.mkdir(exist_ok=True)

    countries = datalib.get_country_data(datalib.DATA_ROOT)

    for country, df in countries.items():
        train_model(df, country, test)

def train_model(df, country, test=False):

    start_time = time.time()
    dfn = datalib.generate_features(df)

    # remove last 30 days from training
    dfn_train = dfn.iloc[:-30]

    y = dfn_train.pop('revenue')
    X = dfn_train

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    ## train a random forest model
    param_grid_rf = {
        'rf__criterion': ['squared_error','absolute_error'],
        'rf__n_estimators': [10,15,20,25]
    }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                            ('rf', RandomForestRegressor())])

    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    rmse = round(np.sqrt(mean_squared_error(y_test,y_pred)))
    r2 = r2_score(y_test, y_pred)

    # save model
    model_name = f'{country}.joblib' if not test else f'test-{country}.joblib'
    joblib.dump(grid, MODEL_ROOT / model_name)
    logger.log_train(model_name, X_train.shape, {'rmse': rmse, 'r2':r2}, 
        time.time() - start_time, MODEL_VERSION, MODEL_VERSION_NOTE)

def load_model(country):
    """
    Load model and data
    """

    # find and load country specific model
    if not (MODEL_ROOT / f'{country}.joblib').exists():
        raise Exception('Model file not found')

    model = joblib.load(MODEL_ROOT / f'{country}.joblib')

    data = datalib.get_country_data(datalib.DATA_ROOT, country=country)

    return model, data


def predict_model(country, year, month, day):
    """
    Get data and model
    Return prediction result
    """
    start_time = time.time()
    model, df = load_model(country)

    # check if date in dataset
    date = datetime.datetime(int(year), int(month), int(day))
    if date not in df.index:
        raise Exception('Date not in dataset')
    
    dfn = datalib.generate_features(df)

    y = dfn.pop('revenue')
    X = dfn.loc[date:date]

    y_pred = model.predict(X)

    logger.log_pred(list(y_pred), [country, year, month, day], 
        time.time() - start_time, MODEL_VERSION)

    return { 'y_pred': list(y_pred) }


if __name__ == "__main__":

    # fetch data from ingest_data
    countries = datalib.get_country_data()
