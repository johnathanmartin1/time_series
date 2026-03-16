# -*- coding: utf-8 -*-
"""
Created on Thu March  12 13:07:24 2026

@author: John Martin

"""
"""_________________________________________________________________________"""

"""____________________IMPORTING THE REQUIRED PACKAGES______________________"""


import numpy as np
from itertools import product
from typing import Tuple
import matplotlib.pyplot as plt
#from src.acf.acf_cpp_backend import acf_cpp_backend_calc
#from src.tools.confidence_interval import confidence_interval_plot
#from typing import Literal

def MA_builder(x:float) -> float:
    return 2 + 1 * np.random.normal(0,0.1) #- 1.2 * np.random.normal(0.5,1.4)

def MA_constructor() -> np.array:
    MAmodel = []
    for i in range(100):
        MAmodel.append(MA_builder(MAmodel))
    return np.array(MAmodel)



class MA_model():
    
    def __init__(self, series_data: [list, np.array], theta_q: [list, np.array] = None, mean: float = None, stdev: float = None):
        
        self.theta_q = np.array(theta_q)
        
        self.mean = mean
        
        self.stdev = stdev 
        
        self.series_data = series_data
        
        self.model = None



    def likelihood_py(self, theta: [list, np.array], y: [list, np.array]) -> float :
        
        y = np.array(y)
        
        y_minus_mean_y = y - np.mean(y)
        
        sigma = np.std(y_minus_mean_y)
        
        x = np.array(np.random.normal(0,sigma, size=y.size))
        
        theta_hat = np.array(theta)
        
        y_minus_theta_x = y_minus_mean_y
        
        for i in theta_hat:
            
            y_minus_theta_x -= i*x
        
        gaussian_probability = (1/np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(y_minus_theta_x)**2 / (2 * sigma**2))
        
        return np.sum(np.log(gaussian_probability))


    
    def fit(self, q: int, min_theta_q: float = -1,
               max_theta_q: float = 1, step_theta_q: float =0.01):
        
        prob_dict = {}
        
        q_range = np.arange(min_theta_q, max_theta_q+step_theta_q , step_theta_q)
        
        q_values = np.tile(q_range, (q,1))
        
        for combination in product(*q_values):
        
            prob_dict[combination] = self.likelihood_py(combination, self.series_data)
            
        max_key = max(prob_dict, key=prob_dict.get)
        
        self.theta_q = np.array(max_key)
        
        self.mean = np.mean(self.series_data)
        
        self.stdev = np.std(self.series_data)

        
    
    def model_function(self, decimal_places: int = 4):
        
        if self.mean != None and self.theta_q.all() != None and self.stdev != None:
        
            numerical_model_temp = f"{self.mean:.{decimal_places}f} + \u03B5_t"
    
            for lag, coefficient in enumerate(self.theta_q):
            
                lag += 1 
                
                if coefficient < 0:
                
                    coefficient = abs(coefficient)
                    
                    numerical_model_temp += f" - {coefficient:.{decimal_places}f}\u03B5_(t-{lag})"
                
                else:
                
                    numerical_model_temp += f" + {coefficient:.{decimal_places}f}\u03B5_(t-{lag})"
            
            self.model = numerical_model_temp
            
            return self.model
        
        else:
        
            print("MA model not yet fitted, use .fit(q) to fit model first and then call .model_function().")


    
    def numerical_model(self, future_lags:int = 10) -> np.array:
        
        model_time_series = []
        
        for lag in range(0, future_lags):
        
            current_lag_data = 0
            
            for coefficient in self.theta_q:
                
                current_lag_data+= coefficient*np.random.normal(0, self.stdev)
                
            model_time_series.append(current_lag_data + self.mean + np.random.normal(0,self.stdev))
        
        return np.array(model_time_series)


        
    def forecast(self, forecast_lags: int = 50):
        
        lags = [i for i in range(0, len(self.series_data))]
        
        future_lag = [i for i in range(len(self.series_data), len(self.series_data) + forecast_lags)]
        
        forecast_data = self.numerical_model(forecast_lags)
        
        plt.plot(lags, self.series_data, label = "Sample")
        
        plt.plot(future_lag, forecast_data, label = "Forecast")
        
        plt.legend()
        
        plt.xlabel("lags")
        
        plt.show()
        
            



if __name__ == "__main__": 
    data = MA_constructor()
    #model1 = ([0.9,0.5,0.6], 4, 0.1)
    mamodel = MA_model(data) 
      
    mamodel.fit(1)
    
    mamodel.model_function()
    
    print(mamodel.model)
    
    mamodel.forecast(100)
    

    
