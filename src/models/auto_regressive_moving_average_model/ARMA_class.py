# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:23:48 2026

@author: John Martin
"""
import numpy as np
import matplotlib.pyplot as plt


                



class AR_model():
    
    def __init__(self, sample_data: [list, np.array] = None, *, q: int = 1, phi: [list, np.array] = None, mean: float = None, stdev: float = None):
        
        self.sample_data = np.array(sample_data)
        
        self.mean = np.mean(self.sample_data) if mean == None else mean
        
        self.stdev = np.std(self.sample_data) if stdev == None else stdev
        
        self.phi = np.array(phi)
        
        self.model = None
        
        self.error = None
        
        self.forecast_data = None



        
    def model_function(self, decimal_places: int = 4):
        '''Builds a auto regressive model for printing out as a string'''
        if self.mean != None and self.phi.all() != None and self.stdev != None:
        
            numerical_model_temp = f"{self.mean:.{decimal_places}f}"
    
            for lag, coefficient in enumerate(self.phi):
            
                lag += 1 
                
                if coefficient < 0:
                
                    coefficient = abs(coefficient)
                    
                    numerical_model_temp += f" - {coefficient:.{decimal_places}f}Y_(t-{lag})"
                
                else:
                
                    numerical_model_temp += f" + {coefficient:.{decimal_places}f}Y_(t-{lag})"
            
            self.model = numerical_model_temp + " + \u03B5_t"
            
        else:
        
            print("MA model not yet fitted, use .fit(q) to fit model first and then call .model_function().")




    def ar_residuals(self, q: int):
        '''Calculates the reersdiaul errors oft he trial data versus the sample data'''
        
        errors = np.array([0.0]*len(self.sample_data))
        
        for sample_index in range(len(self.sample_data)):
            
            ar_error = 0
            
            for phi_index in range(len(self.phi)):
                
                if sample_index-(phi_index+1) >=0:
                    ar_error += self.phi[phi_index]*self.sample_data[sample_index-(phi_index+1)]
            
            errors[sample_index] = self.sample_data[sample_index] - self.mean - ar_error
            
        return errors
    
    

    
    def loss_function(self, errors):
        '''Calculates the loss function oif the ar_residuals'''
        
        return np.sum(errors**2)
    


    
    def gradient(self,h):
        '''Calculates the gradient of the loss function'''
        
        gradients = np.array([0.0]*len(self.phi))
        
        base_error = self.ar_residuals(len(self.phi))
        
        base_loss = self.loss_function(base_error)
        
        for phi_index in range(len(self.phi)):
            
            self.phi[phi_index] += h
            
            error = self.ar_residuals(len(self.phi))
            
            loss = self.loss_function(error)
            
            gradients[phi_index] = (loss - base_loss) / h
        
        return gradients



    
    def early_loss_convergence(self, loss_bucket):
        '''stops the fitting function early if the past three loss function rersults have converged'''
        
        if len(set(loss_bucket))==1 and len(loss_bucket)==3:
        
            return True
        
        else:
        
            return False
        


    
    def fit(self, q: int, *, lr: float = 0.0001, epochs: int = 2001, h:float = 1e-6, decimal_places: int = 4):
        '''Fits the AR function based upon th echosen number of lags
                - q is the chosen number of lags
                - lr is the learening rate (default = 0.0001)
                - epochs is the number of iterations the function will perform to fiund teh miniumum loss function (default = 2000)
                - h is the step size of the nuumerical gradient smaller is more refined (default = 1e-6)
                - decimal_places is accuracy of the to determine the loss function to less decimal places are less accurate (default = 4)'''
                
        self.phi = np.array([0.0]*q)
        
        loss_bucket=[]
        
        for epoch in range(epochs+1):
            
            self.error = self.ar_residuals(q)
            
            loss = self.loss_function(self.error)
            
            gradients = self.gradient(h)
            
            for phi_index in range(q):
                
                self.phi[phi_index] -= lr* gradients[phi_index]
            
            loss_bucket.append(loss.round(decimal_places))
            
            if len(loss_bucket) > 3:
                loss_bucket.pop(0)
            
            if self.early_loss_convergence(loss_bucket) == True:
                print(f"Epoch: {epoch}, loss: {loss:.{decimal_places}f}")
                break
            
            if epoch % 50 == 0:
                print(f"Epoch: {epoch}, loss: {loss:.{decimal_places}f}")
        
        self.mopdel = self.model_function(decimal_places)
        
        
        
                
    def forecast(self, forecast_data_points: int = 10, *, y_axis: str = "Value"):
        '''Plots a forecast of the model aftere fitting
                - forecast_data_points is the number of future data points that are required (default = 10)
                - yaxis can be changed depending on what the data set represents'''
                
        lags  = [i for i in range(0, len(self.sample_data))]
        
        future_lags = [ i for i in range(len(self.sample_data), len(self.sample_data) + forecast_data_points)]
        
        self.forecast_data = [i for i in self.sample_data[-len(self.phi):]]
        
        for loop in range(forecast_data_points):
            
            forecast = 0
            
            for index, value in enumerate(self.phi):
                
                forecast += value*self.forecast_data[-index-1]
            
            self.forecast_data.append(forecast)
        
        for _ in range(len(self.phi)):
            
            self.forecast_data.pop(0)
            
        plt.plot(lags, self.sample_data, label="Sample Data", color="blue")
        
        plt.plot(future_lags, self.forecast_data, label = "Forecast", color="green")
        
        plt.legend()
        
        plt.xlabel("Lag")
        
        plt.ylabel(f"{y_axis}", rotation=90)
        
        plt.show()





if __name__ == "__main__":
    
    from time_series import ARMA_constructor
    
    data = ARMA_constructor(1000, q=2, p=3)
    
    plt.plot(range(len(data)), data)
    plt.show()
    
    # armodel = AR_model(data)

    # armodel.fit(2)
    # armodel.model_function()
    
    # print(armodel.model)
    # armodel.forecast(30)
    
    
    # from statsmodels.tsa.arima.model import ARIMA
    
    # model = ARIMA(data, order=(2,0,0))
    
    # model_fit = model.fit()
    # forecast = model_fit.get_forecast(steps=30)
    # print(forecast)
    # plt.plot(range(len(data)), data)
    # plt.plot(range(len(data),len(forecast.predicted_mean)+len(data)), forecast.predicted_mean)
    # print(model_fit.summary())
