# -*- coding: utf-8 -*-
"""
Created on Thu March  12 13:07:24 2026

@author: John Martin

"""
"""_________________________________________________________________________"""

"""____________________IMPORTING THE REQUIRED PACKAGES______________________"""


import numpy as np
import matplotlib.pyplot as plt


class MA_model():
    '''Class that is utilised for fitting a moving average (MA) model'''
    
    def __init__(self, sample_data: [list, np.array] = None, theta: [list, np.array] = None, mean: float = None, stdev: float = None):
        
        self.sample_data = np.array(sample_data)
        
        self.mean = np.mean(self.sample_data) if mean == None else mean
        
        self.stdev = np.std(self.sample_data) if stdev == None else stdev
        
        self.theta = np.array(theta)
        
        self.model = None
        
        self.error = None
   
    

    
    def model_function(self, decimal_places: int = 4):
        '''Builds a model that can be used for printing out a numerical result as a string'''
        
        if self.mean != None and self.theta.all() != None and self.stdev != None:
        
            numerical_model_temp = f"{self.mean:.{decimal_places}f}"
    
            for lag, coefficient in enumerate(self.theta):
            
                lag += 1 
                
                if coefficient < 0:
                
                    coefficient = abs(coefficient)
                    
                    numerical_model_temp += f" - {coefficient:.{decimal_places}f}\u03B5_(t-{lag})"
                
                else:
                
                    numerical_model_temp += f" + {coefficient:.{decimal_places}f}\u03B5_(t-{lag})"
            
            self.model = numerical_model_temp + " + \u03B5_t"
            
        else:
        
            print("MA model not yet fitted, use .fit(q) to fit model first and then call .model_function().")


    
    
    def residuals(self, q: int) -> np.array:
        '''Calculates the residuals of the sample data and the approximate data'''
        
        errors = [0.0]*len(self.sample_data)
        
        for sample_index in range(len(self.sample_data)):
        
            ma_error = 0
            
            for theta_index in range(1,q+1):
                
                if sample_index-theta_index >= 0:
                
                    ma_error += self.theta[theta_index - 1] * errors[sample_index - theta_index] 
                
                errors[sample_index] = self.sample_data[sample_index] - self.mean - ma_error
        
        return np.array(errors)


    

    def loss_function(self, errors: np.array) -> float:
        '''Calculates the loss function based off the residuals'''
        
        return np.sum(errors**2) 
    
    

    
    def early_loss_convergence(self, loss_bucket):
        '''Determines if the past three loss values are the same and will return True, switching of the fitting function early'''
        
        if len(set(loss_bucket))==1 and len(loss_bucket)==3:
        
            return True
        
        else:
        
            return False  
  
    
  
    
    def gradient(self, h:float) -> np.array:
        '''Determines the gradient of the error function'''
      
        gradients = [0.0]*len(self.theta)
        
        base_error = self.residuals(len(self.theta))
        
        base_loss = self.loss_function(base_error)
        
        for theta_index in range(len(self.theta)):
            self.theta[theta_index] += h
            
            error_step = self.residuals(len(self.theta))
            
            loss_step = self.loss_function(error_step)
            
            gradients[theta_index] = (loss_step - base_loss) / h
        
        return np.array(gradients)
        
        
        
        
    def fit(self, q: int, *, lr: float = 0.0001, epochs: int = 2000, h: float =1e-6, decimal_places: int = 4):
        '''Fits the MA function based upon th echosen number of lags
                - q is the chosen number of lags
                - lr is the learening rate (default = 0.0001)
                - epochs is the number of iterations the function will perform to fiund teh miniumum loss function (default = 2000)
                - h is the step size of the nuumerical gradient smaller is more refined (default = 1e-6)
                - decimal_places is accuracy of the to determine the loss function to less decimal places are less accurate (default = 4)'''
                
        self.theta = np.array([0.0] * q)
        
        loss_bucket=[]
        
        for epoch in range(epochs+1):
            
            self.error = self.residuals(q)
            
            loss = self.loss_function(self.error)
            
            gradients = self.gradient(h)
            
            for theta_index in range(q):
                self.theta[theta_index] -= lr * gradients[theta_index]
            
            loss_bucket.append(loss.round(decimal_places))
            
            if len(loss_bucket) > 3:
                loss_bucket.pop(0)
            
            if self.early_loss_convergence(loss_bucket) == True:
                print(f"Epoch: {epoch}, loss: {loss:.{decimal_places}f}")
                break
                
            if epoch % 50 == 0:
                print(f"Epoch: {epoch}, loss: {loss:.4f}")
        
        
        self.model_function(decimal_places)
        
            
            
            
    def forecast(self, forecast_data_points=10):
        '''Plots a forecast of the model aftere fitting
                - forecast_data_points is the number of future data points that are required (default = 10)'''
                
        lags  = [i for i in range(0, len(self.sample_data))]
        
        future_lags = [ i for i in range(len(self.sample_data), len(self.sample_data) + forecast_data_points)]
        
        forecast_data = []
        
        for i in reversed(range(-1*forecast_data_points+len(self.theta)+1,len(self.theta)+1)):
            
            if i<=0:
            
                forecast_data.append(self.mean)
        
            else:
        
                forecast_data.append(self.mean + np.sum(self.theta[-i:] * self.error[-i:]))
        
        plt.plot(lags, self.sample_data, label="Sample Data", color="blue")
        
        plt.plot(future_lags, forecast_data, label = "Forecast", color="green")
        
        plt.legend()
        
        plt.xlabel("Lag")
        
        plt.show()
        
            
        
                    
        
            



if __name__ == "__main__": 
    
    from time_series import MA_constructor
    
    data = MA_constructor(100, q=6)
    
    mamodel = MA_model(data) 
    
    mamodel.fit(2)

    print(mamodel.model)
    
    mamodel.forecast(100)
    
    
    


    

    
