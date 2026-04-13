# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:06:50 2026

@author: johna
"""

from src.models.model_constructors.stationality_functions import stationality_phi
import numpy as np

def AR_constructor(length:int = 100, *, p: int = 1, phi: int = None, mean: float = 0, stdev: float = 1):
    '''Constructs an auto regrtessive (AR) time series. 
            - length sets the number of data points in the time series (default = 100)
            - p sets the number of past lags to use when building the model and will give random phi values (default = 1)
            - phi will override the p term and produce a model using the specific constants for lags that have been given in a list (default = None)
            - stdev will set the standard deviation for the model (default = 1)
            - mean will set the average of the model (default = 0)'''
    
    if phi == None and p <=0 or length <= 0 or  stdev <=0:
        
        print("Cannot build Auto Regressive model with the chosen inputs.")
        
        print("Check that phi is a list with a least one element or that p is an integer greater than 0.")
        
        print("Length of model data must be greater than 0 and standard deviation (stdev) must be greater than 0.")
    
    else:
        
        if phi == None:
            
            phi = [np.random.uniform(0,1) for _ in range(p)]
        
        else:
            
            if stationality_phi(phi) == False:
                print("Chosen auto regressive paramters do not make a stationary auto regressive tmine series model.")
                return
        
        while stationality_phi(phi) == False:
            
            phi = [np.random.uniform(0,1) for _ in range(p)]
        
        print("AR sample data parameters: phi =", phi)
        
        print("AR sample stationality =", stationality_phi(phi))
        
        armodel = np.array([0.0]*length)
    
        armodel[0] = np.random.normal(mean,stdev)
    
        for model_i in range(1,length):
            
            ar_component = 0
            
            for phi_i in range(len(phi)):
                
                if model_i-(phi_i+1) >=0:
                    
                    ar_component += phi[phi_i]*armodel[model_i-(phi_i+1)]
           
            armodel[model_i] = mean + ar_component + np.random.normal(0, stdev)
        
        return armodel