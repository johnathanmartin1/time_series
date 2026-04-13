# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:05:35 2026

@author: John Martin
"""
from src.models.model_constructors.stationality_functions import stationality_theta
import numpy as np

def MA_constructor(length: int = 100, *, q: int = 1, theta: list = None, stdev: float = 1, mean: float = 0.0) -> np.array:
    '''Constructs a moving average (MA) time series. 
            - length sets the number of data points in the time series (default = 100)
            - q sets the number of past lags to use when building the model and will give random theta values (default = 1)
            - theta will ovewrride the q term and produce a model using the specific constants for lags that have been given in a list (default = None)
            - stdev will set the standard deviation for the model (default = 1)
            - mean will set the average of the model (default = 0)'''
    
    if theta == None and q <=0 or length <= 0 or  stdev <=0:
        print("Cannot build Moving Average model with the chosen inputs.")
        print("Check that theta is a list with a least one element or that q is an integer greater than 0.")
        print("Length of model data must be greater than 0 and standard deviation (stdev) must be greater than 0.")
        
    else:
        if theta == None:
            
            theta = [np.random.uniform(0,1) for _ in range(q)]
        
        else:
            
            if stationality_theta(theta) == False:
                print("Chosen moving average paramters (theta) do not make a stationary moving average time series model.")
                return
        
        while stationality_theta(theta) == False:
            
            theta  = [np.random.uniform(0,1) for _ in range(q)]
        
        MAmodel = np.array([0.0]*length)
        
        for i in range(length):
            
            ma_noise = 0
            
            for thet in theta:
            
                ma_noise += thet*np.random.normal(0,stdev)  
            
            MAmodel[i] = mean + ma_noise + np.random.normal(0,stdev)
        
        return MAmodel
