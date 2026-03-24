# -*- coding: utf-8 -*-
"""
Created on Thu March  12 13:07:24 2026

@author: John Martin

"""
"""_________________________________________________________________________"""

"""____________________IMPORTING THE REQUIRED PACKAGES______________________"""


import numpy as np
import matplotlib.pyplot as plt





def MA_constructor(length: int = 100, *, q: int = 1, theta: list = None, stdev: float = 1, mean: float = 0.0) -> np.array:
    
    if theta == None and q <=0 or length <= 0 or  stdev <=0:
        print("Cannot build Moving Average model with the chosen inputs.")
        print("Check that theta is a list with a least one element or that q is an integer greater than 0.")
        print("Length of model data must be greater than 0 and standard deviation (stdev) must be greater than 0.")
        
    else:
        if theta == None:
            
            theta = [np.random.uniform(-1,1) for _ in range(q)]
        print(theta)
        
        MAmodel = np.array([0.0]*length)
        
        for i in range(length):
            
            ma_noise = 0
            
            for thet in theta:
            
                ma_noise += thet*np.random.normal(0,stdev)  
            
            MAmodel[i] = mean + ma_noise
        
        return MAmodel



class MA_model():
    
    def __init__(self, sample_data: [list, np.array] = None, theta: [list, np.array] = None, mean: float = None, stdev: float = None):
        
        self.sample_data = np.array(sample_data)
        
        self.mean = np.mean(self.sample_data) if mean == None else mean
        
        self.stdev = np.std(self.sample_data) if stdev == None else stdev
        
        self.theta = np.array(theta)
        
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
        
        
        
        
        
    
    '''fitting the theta weightings with gradient descent and assigning the last q errors for forecasting'''
    def fit(self, q: int, *, lr = 0.0001, epochs = 2000, h=1e-6):
        
        self.theta = np.array([0.5] * q)
        
        for epoch in range(epochs):
            
            self.error = self.residuals(q)
            
            loss = self.loss_function(self.error)
            
            gradients = self.gradient(h)
            
            for theta_index in range(q):
                self.theta[theta_index] -= lr * gradients[theta_index]
                
            if epoch % 50 == 0:
                print(f"Epoch: {epoch}, loss: {loss:.4f}")
        
        
        self.model_function()
                
        
            
            
            
            
    '''Plotting a forecast of the model'''   
    def forecast(self, steps=10):
        
        lags  = [i for i in range(0, len(self.sample_data))]
        
        future_steps = [ i for i in range(len(self.sample_data), len(self.sample_data) + steps)]
        
        forecast_data = []
        
        for i in reversed(range(-1*steps+len(self.theta)+1,len(self.theta)+1)):
            
            if i<=0:
            
                forecast_data.append(self.mean)
        
            else:
        
                forecast_data.append(self.mean + np.sum(self.theta[-i:] * self.error[-i:]))
        
        plt.plot(lags, self.sample_data, label="Sample Data", color="blue")
        
        plt.plot(future_steps, forecast_data, label = "Forecast", color="green")
        
        plt.legend()
        
        plt.xlabel("Lag")
        
        plt.show()
        
            
        
                    
        
            



if __name__ == "__main__": 
    
    data = MA_constructor(100, q=6)
    
    mamodel = MA_model(data) 
    
    mamodel.fit(6)

    print(mamodel.model)
    
    mamodel.forecast(100)
    
    
    


    

    
