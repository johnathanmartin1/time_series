# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:44:22 2026

@author: John Martin
"""

import numpy as np
import matplotlib.pyplot as plt

def stationality(phi):
    
    if len(phi)== 1:
    
        return True if -1<=phi[0]<=1 else False
    
    else:
        parameter = []
        
        for i in phi:
            parameter.append(-i)
        
        parameter.append(1)
        
        roots = np.roots(parameter)
       
        return all([abs(r)>1 for r in roots])



def AR_constructor(length:int = 100, *, q: int = 1, phi: int = None, mean: float = 0, stdev: float = 1):
    
    if phi == None and q <=0 or length <= 0 or  stdev <=0:
        
        print("Cannot build Moving Average model with the chosen inputs.")
        
        print("Check that phi is a list with a least one element or that q is an integer greater than 0.")
        
        print("Length of model data must be greater than 0 and standard deviation (stdev) must be greater than 0.")
    
    else:
        
        if phi == None:
            
            phi = [np.random.uniform(0,1) for _ in range(q)]
        
        else:
            
            if stationality(phi) == False:
                print("Chosen auto regressive paramters do not make a stationary auto regressive tmine series model.")
                return
        
        while stationality(phi) == False:
            
            phi = [np.random.uniform(0,1) for _ in range(q)]
        
        print("AR sample data parameters: phi =", phi)
        
        print("AR sample stationality =", stationality(phi))
        
        armodel = np.array([0.0]*length)
    
        armodel[0] = np.random.normal(mean,stdev)
    
        for model_i in range(1,length):
            ar_component = 0
            
            for phi_i in range(len(phi)):
                
                if model_i-(phi_i+1) >=0:
                    
                    ar_component += phi[phi_i]*armodel[model_i-(phi_i+1)]
           
            armodel[model_i] = mean + ar_component + np.random.normal(0, stdev)
        
        return armodel
                



class AR_model():
    
    def __init__(self, sample_data: [list, np.array] = None, *, q: int = 1, phi: [list, np.array] = None, mean: float = None, stdev: float = None):
        
        self.sample_data = np.array(sample_data)
        
        self.mean = np.mean(self.sample_data) if mean == None else mean
        
        self.stdev = np.std(self.sample_data) if stdev == None else stdev
        
        self.phi = np.array(phi)
        
        self.model = None
        
        self.error = None

    
    def ar_residuals(self, q: int):
        
        errors = np.array([0.0]*len(self.sample_data))
        
        for sample_index in range(len(self.sample_data)):
            
            ar_error = 0
            
            for phi_index in range(len(self.phi)):
                
                if sample_index-(phi_index+1) >=0:
                    ar_error += self.phi[phi_index]*self.sample_data[sample_index-(phi_index+1)]
            
            errors[sample_index] = self.sample_data[sample_index] - self.mean - ar_error
            
        return errors
    
    
    
    def loss_function(self, errors):
        
        return np.sum(errors**2)
    
    
    def gradient(self,h):
        
        gradients = np.array([0.0]*len(self.phi))
        
        base_error = self.ar_residuals(len(self.phi))
        
        base_loss = self.loss_function(base_error)
        
        for phi_index in range(len(self.phi)):
            
            self.phi[phi_index] += h
            
            error = self.ar_residuals(len(self.phi))
            
            loss = self.loss_function(error)
            
            gradients[phi_index] = (loss - base_loss) / h
        
        return gradients
    
    
    
    def fit(self, q: int, *, lr: float = 0.0001, epochs: int = 2001, h:float = 1e-6):
        
        self.phi = np.array([0.0]*q)
        
        for epoch in range(epochs+1):
            
            self.error = self.ar_residuals(q)
            
            loss = self.loss_function(self.error)
            
            gradients = self.gradient(h)
            
            for phi_index in range(q):
                
                self.phi[phi_index] -= lr* gradients[phi_index]
            
            if epoch % 50 == 0:
                print(f"Epoch: {epoch}, loss: {loss:.4f}")
        
        
        
        
        


if __name__ == "__main__":
    
    data = AR_constructor(1000, q=4)
    
    plt.plot(range(len(data)), data)
    plt.show()
    
    armodel = AR_model(data)

    armodel.fit(4)
    print(armodel.phi)
    
    from statsmodels.tsa.arima.model import ARIMA
    
    model = ARIMA(data, order=(4,0,0))
    
    model_fit = model.fit()
    
    print(model_fit.summary())