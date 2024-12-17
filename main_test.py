"""
Example of Implementing a Localized ML forecasting model 
(training and testing on a station named SCA)

This script demonstrates how to create and test a localized machine learning 
forecasting model using weather data from a single station.

Author: Alberto Ardid
Email: aardids@gmail.com
Version: 0.1.0
"""

from datetime import timedelta
from fire.model import ForecastModel, MultiStationForecastModel
from fire.data import WeatherData, GeneralData
from fire.utilities import datetimeify, load_dataframe
from glob import glob
import pandas as pd
import numpy as np
import os, shutil, json, pickle, csv
import matplotlib.pyplot as plt

# Define time constants for easier date handling
_MONTH = timedelta(days=365.05 / 12)
_DAY = timedelta(days=1)
n_jobs = 4  # Number of parallel processing jobs

def forecast_test():
    """
    Test the forecast model by training and testing on the SCA station.

    This function initializes the data streams, sets up the MultiStationForecastModel,
    trains the model, and performs high-resolution forecasting for a specific wildfire event.
    """
    # Define data ranges for the SCA station
    data = {'SCA': ['2002-03-01', '2009-12-01']}
    
    # Define wildfire indices for the SCA station
    eruptions = {'SCA': [i for i in range(0, 8)]}
    
    # Define weather data streams to use in the model
    data_streams = ['T', 'V', 'RH']  # Temperature, Wind Speed, Relative Humidity
    
    # Create a MultiStationForecastModel instance
    fm = MultiStationForecastModel(
        data=data,               # Station data ranges
        window=2.0,              # Forecasting window (days)
        overlap=0.75,            # Overlap fraction between windows
        look_forward=2.0,        # Look-forward period (days)
        data_streams=data_streams,  # Weather data streams
        root='test_SCA'          # Directory for saving results
    )
    
    # Define features to drop during training
    drop_features = ['linear_trend_timewise', 'agg_linear_trend']
    
    # Compute forecast over the wildfire period
    te = fm.data['SCA'].fis[2]  # Use the third wildfire event
    ti = te - _DAY * 10  # Start time: 10 days before wildfire
    tf = te + _DAY * 10  # End time: 10 days after wildfire
    
    # Exclude data around wildfires to avoid overfitting
    exclude_dates = {}
    for _sta in data.keys():  # Initialize exclusions for all stations
        exclude_dates[_sta] = None
    exclude_dates['SCA'] = [[ti, tf]]  # Exclude data for SCA during the wildfire period
    
    # Train the model
    fm.train(
        drop_features=drop_features,  # Features to exclude
        retrain=True,                 # Retrain the model
        Ncl=150,                      # Number of clusters for feature extraction
        n_jobs=n_jobs,                # Parallel processing jobs
        exclude_dates=exclude_dates   # Dates to exclude for training
    )
    
    # Perform high-resolution forecasting for SCA
    fm.hires_forecast(
        station='SCA',          # Station to forecast
        ti=ti,                  # Forecast start time
        tf=tf,                  # Forecast end time
        recalculate=True,       # Use cached results if available
        n_jobs=n_jobs,          # Parallel processing jobs
        threshold=0.75,         # Threshold for wildfire probabilities
        root='test_SCA',        # Root directory for saving results
        save='plots' + os.sep + 'test_SCA' + os.sep + '_fc_eruption_' + 'SCA' + '.png'
                                # Path to save the forecast plot
    )

def main():
    """
    Main function to run the forecast test.
    """
    forecast_test()  # Run the forecast test
    pass  # Placeholder for additional functionality

if __name__ == '__main__':
    main()  # Execute the script