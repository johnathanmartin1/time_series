# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:44:22 2026

@author: John Martin
"""

import numpy as np
import matplotlib.pyplot as plt

def stationality(theta):
    
    if len(theta)== 1:
    
        return True if -1<=theta[0]<=1 else False
    
    else:
        parameter = []
        
        for i in theta:
            parameter.append(-i)
        
        parameter.append(1)
        
        roots = np.roots(parameter)
       
        return all([abs(r)>1 for r in roots])



def AR_constructor(length:int = 100, *, q: int = 1, theta: int = None, mean: float = 0, stdev: float = 1):
    
    if theta == None and q <=0 or length <= 0 or  stdev <=0:
        
        print("Cannot build Moving Average model with the chosen inputs.")
        
        print("Check that theta is a list with a least one element or that q is an integer greater than 0.")
        
        print("Length of model data must be greater than 0 and standard deviation (stdev) must be greater than 0.")
    
    else:
        
        if theta == None:
            
            theta = [np.random.uniform(0,1) for _ in range(q)]
        
        else:
            
            if stationality(theta) == False:
                print("Chosen auto regressive paramters do not make a stationary auto regressive tmine series model.")
                return
        
        while stationality(theta) == False:
            
            theta = [np.random.uniform(0,1) for _ in range(q)]
        print(theta)
        print(stationality(theta))
        armodel = np.array([0.0]*length)
    
        armodel[0] = np.random.normal(mean,stdev)
    
        for model_i in range(1,length):
            ar_component = 0
            
            for theta_i in range(len(theta)):
                
                if model_i-(theta_i+1) >=0:
                    
                    ar_component += theta[theta_i]*armodel[model_i-(theta_i+1)]
           
            armodel[model_i] = mean + ar_component + np.random.normal(0, stdev)
        
        return armodel
                



class AR_model():
    
    def __init__(self, sample_data: [list, np.array] = None, *, q: int = 1, theta: [list, np.array] = None, mean: float = None, stdev: float = None):
        
        self.sample_data = np.array(sample_data)
        
        self.mean = np.mean(self.sample_data) if mean == None else mean
        
        self.stdev = np.std(self.sample_data) if stdev == None else stdev
        
        self.theta = np.array(theta)
        
        self.model = None
        
        self.error = None

    def ar_residuals(self, q: int):
        
        errors = np.array([0.0]*len(self.sample_data))
        
        for sample_index in range(len(self.sample_data)):
            
            ar_error = 0
            
            for theta_index in range(len(self.theta)):
                
                if sample_index-(theta_index+1) >=0:
                    ar_error += self.theta[theta_index]*self.sample_data[sample_index-(theta_index+1)]
            
            errors[sample_index] = self.sample_data[sample_index] - self.mean - ar_error
            
        return errors
        
        


if __name__ == "__main__":
    
    data = AR_constructor(1000, q=3)
    #print(data)
    plt.plot(range(len(data)), data)
    plt.show()
    
    armodel = AR_model(data, q=2)
    armodel.theta=[0.1,0.2]
    print(armodel.ar_residuals(2))