# -*- coding: utf-8 -*-
"""
Created on Thu March  12 13:07:24 2026

@author: John Martin

"""
"""_________________________________________________________________________"""

"""____________________IMPORTING THE REQUIRED PACKAGES______________________"""


import numpy as np
import matplotlib.pyplot as plt
#from src.acf.acf_cpp_backend import acf_cpp_backend_calc
#from src.tools.confidence_interval import confidence_interval_plot
#from typing import Literal



def MA_constructor(length: int = 100, *, q: int = 1, theta: list = None, stdev: float = 1, mean: float = 0.0) -> np.array:
    
    if theta == None and q <=0 or length <= 0 or  stdev <=0:
        print("Cannot build Moving Average model with the chosen inputs.")
        print("Check that theta is a list with a least one element or that q is an integer greater than 0.")
        print("Length of model data must be greater than 0 and standard deviation (stdev) must be greater than 0.")
        
    else:
        if theta == None:
            
            theta = [np.random.uniform(-1,1) for _ in range(q)]
        print(theta)
        error = [np.random.normal(0,stdev) for _ in theta]
        
        MAmodel = np.array([0.0]*length)
        
        for i in range(length):
        
            ma_construct = 0
            
            for j in range(len(theta)):
            
                if i-j >=0:
                
                    ma_construct += theta[j] * error[j]
            
            MAmodel[i] = mean + np.random.normal(0,stdev) + ma_construct
            
        return MAmodel



class MA_model():
    
    def __init__(self, sample_data: [list, np.array] = None, theta: [list, np.array] = None, mean: float = None, stdev: float = None):
        
        self.sample_data = np.array(sample_data)
        
        self.mean = np.mean(self.sample_data)
        
        self.stdev = np.std(self.sample_data)
        
        self.theta = np.array(theta)
        
        self.lags = None
        
        self.model = None
        
        self.error = None
   
    '''builds a model for printing'''
    def model_function(self, decimal_places: int = 4):
        
        if self.mean != None and self.theta.all() != None and self.stdev != None:
        
            numerical_model_temp = f"{self.mean:.{decimal_places}f} + \u03B5_t"
    
            for lag, coefficient in enumerate(self.theta):
            
                lag += 1 
                
                if coefficient < 0:
                
                    coefficient = abs(coefficient)
                    
                    numerical_model_temp += f" - {coefficient:.{decimal_places}f}\u03B5_(t-{lag})"
                
                else:
                
                    numerical_model_temp += f" + {coefficient:.{decimal_places}f}\u03B5_(t-{lag})"
            
            self.model = numerical_model_temp
            
        
        else:
        
            print("MA model not yet fitted, use .fit(q) to fit model first and then call .model_function().")


    
    '''function that calculates the residuals between the sample data and the predicted data'''
    def residuals(self, q: int) -> np.array:
        
        errors = [0.0]*len(self.sample_data)
        
        for sample_index in range(len(self.sample_data)):
        
            ma_error = 0
            
            for theta_index in range(1,q+1):
                
                if sample_index-theta_index >= 0:
                
                    ma_error += self.theta[theta_index - 1] * errors[sample_index - theta_index] 
                
                errors[sample_index] = self.sample_data[sample_index] - self.mean - ma_error
        
        return np.array(errors)

    

    def loss_function(self, errors: np.array) -> float:
        
        return np.sum(errors**2)
     
       
  
    def gradient(self, h:float) -> np.array:
        
        gradients = [0.0]*len(self.theta)
        
        base_error = self.residuals(len(self.theta))
        
        base_loss = self.loss_function(base_error)
        
        for theta_index in range(len(self.theta)):
            self.theta[theta_index] += h
            
            error_step = self.residuals(len(self.theta))
            
            loss_step = self.loss_function(error_step)
            
            gradients[theta_index] = (loss_step - base_loss) / h
        
        return np.array(gradients)
        
        
        
        
        
    
    '''fittinmg the theta weightings with gradient descent'''
    def fit(self, q: int, lr = 0.00001, epochs = 10000, h=1e-6):
        
        self.theta = np.array([0.0] * q)
        
        for epoch in range(epochs):
            
            error = self.residuals(q)
            
            loss = self.loss_function(error)
            
            gradients = self.gradient(h)
            
            for theta_index in range(q):
                self.theta[theta_index] -= lr * gradients[theta_index]
                
            if epoch % 50 == 0:
                print(f"Epoch: {epoch}, loss: {loss:.4f}")
        
        self.error = error[-q:]
        self.model_function()
                
        
            
            
            
            
        
        
        
        
                    
        
        
        
        
    
    
        
    # def forecast(self, forecast_lags: int = 50):
        
    #     lags = [i for i in range(0, len(self.series_data))]
        
    #     future_lag = [i for i in range(len(self.series_data), len(self.series_data) + forecast_lags)]
        
    #     forecast_data = self.numerical_model(forecast_lags)
        
    #     plt.plot(lags, self.series_data, label = "Sample")
        
    #     plt.plot(future_lag, forecast_data, label = "Forecast")
        
    #     plt.legend()
        
    #     plt.xlabel("lags")
        
    #     plt.show()
        
            



if __name__ == "__main__": 
    data = MA_constructor(q=5)
    plt.plot(range(len(data)), data)
    plt.show()
    #model1 = ([0.9,0.5,0.6], 4, 0.1)
    mamodel = MA_model(data) 
    
    mamodel.fit(5)
    # print(mamodel.theta)
    # mamodel.model_function()
    print(mamodel.model)
    print(mamodel.error)
    # mamodel1 =MA_model(data, [0.9,0.1])
    # mamodel1.model_function()
    # print(mamodel1.model)
    # print(np.sum(mamodel.theta))
    
    


    

    
