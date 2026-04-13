# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:09:27 2026

@author: John Martin
"""

from src.models.model_constructors.stationality_functions import stationality_phi, stationality_theta
import numpy as np

def input_check_arma(length, q, phi, p,  theta, stdev):
    if phi == None and q <=0:
        
        print("Cannot build Auto Regressive Moving Average model with the chosen inputs.")
        
        print("Check that phi is a list with a least one element or that q is an integer greater than 0.")
        
    if p <= 0 and theta == None:
        
        print("Cannot build Auto Regressive Moving Average model with the chosen inputs.")
        
        print("Check that theta is a list with at least one float or that p is an integer greater than 0")
    
    if length <= 0 or  stdev <=0:
        
        print("Cannot build Auto Regressive Moving Average model with the chosen inputs.")
        
        print("Length of model data must be greater than 0 and standard deviation (stdev) must be greater than 0.")
    
    else:
        return True
        
       
        
        
def ARMA_constructor(length:int = 100, *, p: int = 1, phi: list = None, q: int = 1, theta: list = None, mean: float = 0, stdev: float = 1):
    '''Constructs a auto regressive moving average (ARMA) time series. 
            - length sets the number of data points in the time series (default = 100)
            - p sets the number of past lags to use when building the AR part of the model and will give random phi values (default = 1)
            - phi will ovewride the q term and produce a model using the specific constants for lags that have been given in a list (default = None)
            - q sets the number of past lags to use when building the MA part of the model and will give random theta values (default = 1)
            - theta will ovewride the q term and produce a model using the specific constants for lags that have been given in a list (default = None)
            - stdev will set the standard deviation for the model (default = 1)
            - mean will set the average of the model (default = 0)'''

    
    if input_check_arma(length, p, phi, q, theta, stdev) == True:
        
        if phi == None:
            
            phi = [np.random.uniform(0,1) for _ in range(p)]
        
        else:
            
            if stationality_phi(phi) == False:
                print("Chosen auto regressive paramters (phi) do not make a stationary auto regressive time series model.")
                return
        
        if theta == None:
            
            theta = [np.random.uniform(0,1) for _ in range(q)]
        
        else:
            
            if stationality_theta(theta) == False:
                print("Chosen moving average paramters (theta) do not make a stationary moving average time series model.")
                return
        
        while stationality_phi(phi) == False:
            
            phi = [np.random.uniform(0,1) for _ in range(p)]
            
        while stationality_theta(theta) == False:
            
            theta  = [np.random.uniform(0,1) for _ in range(q)]
        
        print("AR sample data parameters: phi =", phi)
        
        print("AR sample stationality =", stationality_phi(phi))
        
        print("MA sample_data parameters: theta =", theta)
        
        print("MA samnple stationality =", stationality_theta(theta))
        
        armamodel = np.array([0.0]*length)
    
        armamodel[0] = np.random.normal(mean,stdev)
    
        for model_i in range(1,length):
            
            ar_component = 0
            
            for phi_i in range(len(phi)):
                
                if model_i-(phi_i+1) >=0:
                    
                    ar_component += phi[phi_i]*armamodel[model_i-(phi_i+1)]
            
            ma_component = 0
            
            for thet in theta:
            
                ma_component += thet*np.random.normal(0,stdev) 
           
            armamodel[model_i] = mean + ar_component + ma_component + np.random.normal(0, stdev)
        
        return armamodel