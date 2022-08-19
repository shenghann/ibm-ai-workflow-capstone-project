import csv
from itertools import count
import os
import glob
import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Data ingestion script
# read from json
# output country-level CSVs

DATA_ROOT = Path('data')
RAW_DATA_ROOT = DATA_ROOT / 'cs-train'

def ingest_data(dir):
    """
    Load json files from directory path
    Renames key columns for known unstandardized namings
    """

    # check valid paths and has files
    if not dir.is_dir():
        raise Exception('Path is not a directory')
    if not dir.exists():
        raise Exception('Directory does not exist')
    if len(list(dir.glob('*.json'))) == 0:
        raise Exception('No JSON files in directory')

    # read json files into single dataframe and rename columns for into standard naming
    data_list = []
    for json_file in dir.glob('*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            data_df = pd.DataFrame(data)
            data_df.rename(columns={
                'StreamID':'stream_id',
                'TimesViewed':'times_viewed',
                'total_price':'price'},inplace=True)
            data_list.append(data_df)

    df = pd.concat(data_list)

    # combine year month day columns into single datetime datatype column
    df['date'] = df.apply(lambda row: datetime.datetime(int(row.year), int(row.month), int(row.day)),axis=1)

    # remove letters from invoice number
    df.invoice = df.invoice.str.replace('\D+','')

    return df

def process_timeseries(df, country=None):
    """
    Process dataframe and convert into timeseries
    """
    if country:
        df = df[df.country == country]

    # create day range from first ot last day in dataset
    day_range = pd.date_range(df.date.min(), df.date.max())

    # create summarized timeseries dataset for each day
    purchases, invoices, streams, views, revenue = [], [], [], [], []
    for day in day_range:
        rec = df[df.date == day]
        purchases.append(len(rec))
        invoices.append(rec.invoice.nunique())
        streams.append(rec.stream_id.nunique())
        views.append(rec.times_viewed.sum())
        revenue.append(rec.price.sum())
        
    df_daily = pd.DataFrame({'date':day_range,
                            'purchases':purchases,
                            'invoices':invoices,
                            'streams':streams,
                            'views':views,
                            'revenue':revenue})
    df_daily.set_index('date',inplace=True)
    df_daily['year_month'] = df_daily.index.to_period('M')

    return df_daily

def get_country_data(data_dir, country=None):
    """
    Convert dataset to timeseries and split by country
    Return dict of country names to timeseries data
    """

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    if any(data_dir.glob('*.csv')):
        countries = {}
        for csv_file in data_dir.glob('*.csv'):
            print('Reading CSV',csv_file)
            df = pd.read_csv(csv_file)
            df.date = pd.to_datetime(df.date)
            countries[csv_file.stem] = df.set_index('date')
        return countries[country] if country else countries

    df = ingest_data(RAW_DATA_ROOT)

    # filter top 10 countries by revenue
    top_countries = df.groupby('country').price.sum().sort_values(ascending=False)[:10].index
    dft = df[df.country.isin(top_countries)]

    # convert to timeseries
    countries = {}
    countries['all'] = process_timeseries(dft)
    for country_name in top_countries:
        df_daily = process_timeseries(dft, country=country_name)
        countries[country_name] = df_daily
        # save to CSV
        df_daily.to_csv(data_dir / f'{country_name}.csv')
    
    if country:
        return countries[country]
    else:
        return countries

def generate_features(df):

    # create features of revenue
    # for each day in timeseries, create rolling past day, past week, past n day revenue
    features_list = []
    for date, row in df.iterrows():
        past_n_days = [1, 7, 14, 21, 28, 56, 84]
        new_feature = {}
        for past_n in past_n_days:
            new_feature[f'prev_{past_n}'] = df[date-datetime.timedelta(days=past_n):date].revenue.sum()
            
        # label - revenue for the month
        new_feature['revenue'] = df[date:date+datetime.timedelta(days=30)].revenue.sum()
        
        # previous year
        new_feature['prev_year'] = df[date - datetime.timedelta(days=365):date - datetime.timedelta(days=335)].revenue.sum()
        
        new_feature['invoices'] = df[date - datetime.timedelta(days=30):date].invoices.mean()
        new_feature['views'] = df[date - datetime.timedelta(days=30):date].views.mean()
            
        features_list.append(new_feature)

    dfn = pd.DataFrame(features_list, index=df.index)
    return dfn

if __name__ == "__main__":

    # country level timeseries dataset
    countries = get_country_data(DATA_ROOT)