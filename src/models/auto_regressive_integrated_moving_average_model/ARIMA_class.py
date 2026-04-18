# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 01:25:20 2026

@author: John Martin
"""

import numpy as np
import matplotlib.pyplot as plt


class ARIMA_model():
    
    def __init__(self, sample_data: [list, np.array] = None):
        
        self.sample_data = np.array(sample_data)
        
        self.maximum_data = abs(np.max(self.sample_data))
        
        self.sample_data_normalised = self.sample_data.copy()
        
        self.sample_data_diff = [0.0]*len(self.sample_data)
        
        self.mean = None #mean of sample_data_diff
        
        self.stdev = None #standard devaition of sample_data_diff
        
        self.phi = None
        
        self.theta = None
        
        self.integration = 0
        
        self.model = None
        
        self.error = None
        
        self.forecast_data = None

        self.diff_bucket = []


        
    def model_function(self, decimal_places: int = 4) -> None:
        '''Builds an auto regressive moving average model for printing out as a string'''
        
        if self.mean != None and self.phi.all() != None and self.theta.all() != None and self.stdev != None:
        
            numerical_model_temp = f"{self.mean:.{decimal_places}f}"
    
            for lag, coefficient in enumerate(self.phi):
            
                lag += 1 
                
                if coefficient < 0:
                
                    coefficient = abs(coefficient)#*self.maximum_data
                    
                    numerical_model_temp += f" - {coefficient:.{decimal_places}f}Y_(t-{lag})"
                
                else:
                
                    numerical_model_temp += f" + {coefficient:.{decimal_places}f}Y_(t-{lag})"
            
            for lag, coefficient in enumerate(self.theta):
            
                lag += 1 
                
                if coefficient < 0:
                
                    coefficient = abs(coefficient)#*self.maximum_data
                    
                    numerical_model_temp += f" - {coefficient:.{decimal_places}f}\u03B5_(t-{lag})"
                
                else:
                
                    numerical_model_temp += f" + {coefficient:.{decimal_places}f}\u03B5_(t-{lag})"
            
            self.model = numerical_model_temp + " + \u03B5_t"
            
        else:
        
            print("ARMA model not yet fitted, use .fit(p,q) to fit model first and then call .model_function().")




    def residuals(self):
        '''Calculates the reersdiaul errors oft he trial data versus the sample data'''
        
        errors = np.array([0.0]*len(self.sample_data_diff))
        
        for sample_index in range(len(self.sample_data_diff)):
            
            ar_error = 0
            
            for phi_index in range(len(self.phi)):
                
                if sample_index-(phi_index+1) >=0:
                    
                    ar_error += self.phi[phi_index]*self.sample_data_diff[sample_index-(phi_index+1)]
            
            ma_error = 0
            
            for theta_index in range(1,len(self.theta)+1):
                
                if sample_index-theta_index >= 0:
                
                    ma_error += self.theta[theta_index - 1] * errors[sample_index - theta_index] 
                        
            errors[sample_index] = self.sample_data_diff[sample_index] - self.mean - ar_error - ma_error
            
        return errors
    
    
    
    
    def loss_function(self, errors):
        '''Calculates the loss function oif the ar_residuals'''
        
        return np.sum(errors**2)
    


    
    def gradient(self,h):
        '''Calculates the gradient of the loss function'''
        
        gradients = np.array([0.0]*(len(self.phi)+len(self.theta)))
        
        base_error = self.residuals()
        
        base_loss = self.loss_function(base_error)
        
        for phi_index in range(len(self.phi)):
            
            self.phi[phi_index] += h
            
            error = self.residuals()
            
            loss = self.loss_function(error)
            
            gradients[phi_index] = (loss - base_loss) / h
            
        for theta_index in range(len(self.theta)):
            
            self.theta[theta_index] += h
            
            error = self.residuals()
            
            loss = self.loss_function(error)
            
            gradients[theta_index+len(self.phi)] = (loss - base_loss) / h
        
        return gradients



    
    def early_loss_convergence(self, loss_bucket):
        '''stops the fitting function early if the past three loss function rersults have converged'''
        
        if len(set(loss_bucket))==1 and len(loss_bucket)==3:
        
            return True
        
        else:
        
            return False
    
        
    
    def diff(self, i) -> None:
        if self.integration !=0:
            
            sample_data_diff = self.sample_data_normalised.copy()
            
            for _ in range(i):
                
                for index, value in enumerate(range(1,len(sample_data_diff))):
                    
                    sample_data_diff[-index] = sample_data_diff[-index]-sample_data_diff[-index-1]
                
                #sample_data_diff[0] = np.mean(sample_data_diff)
                
                self.diff_bucket.append(sample_data_diff)
                
            self.sample_data_diff = sample_data_diff[:]
             
        else:
            #self.sample_data_diff = [0.0]*len(self.sample_data_normalised)
            
            self.sample_data_diff = self.sample_data_normalised.copy()
        
        self.mean = np.mean(self.sample_data_diff)
        
        self.stdev = np.std(self.sample_data_diff)
        
    
    
    
    def fit(self, p: int, i:int, q: int, *, lr: float = 0.0001, epochs: int = 10000001, h:float = 1e-8, decimal_places: int = 4) -> None:
        '''Fits the AR function based upon th echosen number of lags
                - p is the chosen number of lags for the auto regressive part of the ARMA model
                - q is the chosen number of lags for the moving aberage part of the ARMA model
                - lr is the learening rate (default = 0.0001)
                - epochs is the number of iterations the function will perform to fiund teh miniumum loss function (default = 2000)
                - h is the step size of the nuumerical gradient smaller is more refined (default = 1e-6)
                - decimal_places is accuracy of the to determine the loss function to less decimal places are less accurate (default = 4)'''
        
        if p < 1 and q < 1:
            
            print("Integer values for p or q greater than zero have not been chosen, model fitting aborted")
            
            return
        
        self.integration = i 
        
        self.diff(self.integration)
        
        self.phi = np.array([0.0]*p)
        
        self.theta = np.array([0.0]*q)
        
        
        loss_bucket=[]
        
        for epoch in range(epochs+1):
            
            self.error = self.residuals()
                        
            loss = self.loss_function(self.error)
            
            if loss == np.nan:
                
                print("loss function error, fitting stopped")
            
                break
            
            gradients = self.gradient(h)
            
            for phi_index in range(p):
                
                self.phi[phi_index] -= lr * gradients[phi_index]
            
            for theta_index in range(q):
                
                self.theta[theta_index] -= lr * gradients[theta_index + p]
            
            if epoch >3000:
                loss_bucket.append(loss.round(decimal_places))
            
                if len(loss_bucket) > 3:
                    loss_bucket.pop(0)
                
                if self.early_loss_convergence(loss_bucket) == True:
                    print(f"Epoch: {epoch}, loss: {loss:.{decimal_places}f}")
                    break
            
            if epoch % 50 == 0:
                print(f"Epoch: {epoch}, loss: {loss:.{decimal_places}f}")
        
        self.model_function(decimal_places)
        
        
    def rediff(self, i):
        initial_data=self.forecast_data.copy()
        #if self.integration!=0:
            #initial_data=self.forecast_data.copy()
            
        
        if self.integration >0:
            for index, value in enumerate(initial_data):
                
                if index != 0 and index <= len(self.sample_data_normalised):
                    
                    initial_data[index] +=  self.sample_data[index-1]#+initial_data[index-1] #+ data_diff[index-1]
                
            #elif index != 0 and index > len(self.sample_data_normalised):
                
                # if diff == len(self.diff_bucket)-1:
                    
                #     grad = initial_data[-1]-initial_data[-2]
          
                #     momentum = grad/(len(initial_data)-len(self.sample_data_normalised))**2
                    
                        #initial_data[index] =  initial_data[index-1]   
                    
                      
        
        
        self.forecast_data = np.array(initial_data.copy())
               
       
    # def rediff(self, i):
    #     initial_data=self.forecast_data.copy()
    #     #if self.integration!=0:
    #         #initial_data=self.forecast_data.copy()
            
    #     for diff in range(len(self.diff_bucket)):
            
    #         data_diff = self.diff_bucket[-diff-1].copy()
            
    #         for index, value in enumerate(initial_data):
                
    #             if index != 0 and index <= len(self.sample_data_normalised):
                    
    #                 initial_data[index] +=  self.sample_data[index-1]#+initial_data[index-1] #+ data_diff[index-1]
                    
    #             elif index != 0 and index > len(self.sample_data_normalised):
                    
    #                 if diff == len(self.diff_bucket)-1:
                        
    #                     grad = initial_data[-1]-initial_data[-2]
              
    #                     momentum = grad/(len(initial_data)-len(self.sample_data_normalised))**2
                    
    #                     #initial_data[index] =  initial_data[index-1]   
                    
                      
        
        
    #     self.forecast_data = np.array(initial_data.copy())
        
    def ar_forecast(self, index: int, history: list)-> float:
        '''returns the calculated ar value at the given index'''
        
        ar_forecast = np.mean(self.sample_data)

        for phi_index in range(len(self.phi)):
           
            ar_forecast += self.phi[phi_index]*history[phi_index]
           
        return ar_forecast
     
        
    
    
    def ma_forecast(self, index: int) -> float:
        '''returns the calculated ma value at the given index'''
        
        if len(self.phi)>0:
            
            ma_forecast = 0
        
        else:
        
            ma_forecast=self.mean
    
        if index >= len(self.sample_data):
        
            pass 
            
        else:
        
            for theta_index in range(0, len(self.theta)):
            
                ma_forecast += self.theta[theta_index]*self.error[index-theta_index] 
        
        return ma_forecast
    
    
    
    
    def forecast(self, forecast_data_points: int = 10, *, y_axis: str = "Value"):
        '''Plots a forecast of the model aftere fitting
                - forecast_data_points is the number of future data points that are required (default = 10)
                - yaxis can be changed depending on what the data set represents'''
        
        if self.model == None:
        
            print("Arima model needs fitting to sample data before forecasting.")
            print("Please utilise the .fit method from the arima class.")
            return
        
        lags = range(len(self.sample_data_normalised)+forecast_data_points)
        
        if len(self.phi)>0:
            
            forecast_data=list(self.sample_data_diff[:len(self.phi)])
            
            history=list(self.sample_data_diff[:len(self.phi)])
            
            for index in range(len(self.phi),len(self.sample_data)+forecast_data_points):
                
                forecast_data.append(self.ar_forecast(index, history[-len(self.phi):]))
                
                if index < len(self.sample_data_diff):
                    
                    history.append(self.sample_data_diff[index])
                
                else:
                    
                    for i in forecast_data[-len(self.phi):]:
                        
                        history.append(i)
            
            forecast_data[0]=self.mean
                
        else:
            
            forecast_data=[0.0]*(len(self.sample_data)+forecast_data_points)
        
            forecast_data[0]=self.mean
        
        if len(self.theta) > 0:
            
            for index in range(1,len(self.sample_data)+forecast_data_points):                    
            
                forecast_data[index] += (self.ma_forecast(index))
        
        self.forecast_data = forecast_data.copy()
        
        '''plotting the results against the sample data'''
        plt.plot(lags[:len(self.sample_data)], self.sample_data, label="Sample Data", color="blue")
        
        plt.plot(lags[:], self.forecast_data[:], label = "Forecast", color="green")
        
        plt.legend()
        
        plt.xlabel("Lag")
        
        plt.ylabel(f"{y_axis}", rotation=90)
        
        plt.show()  
        
       





if __name__ == "__main__":
    
    from time_series import ARIMA_constructor
    p=1
    q=1
    i=0
    data = ARIMA_constructor(100, p=p, i=i, q=q, p_seed=None, q_seed=None)
    
    plt.plot(range(len(data)), data)
    plt.show()
    
    arima_model = ARIMA_model(data)
    
    
    arima_model.fit(p,i,q, decimal_places=4)
    
    # plt.plot(range(len(arima_model.sample_data_diff)), arima_model.sample_data_diff)
    # plt.show()

    print(arima_model.model)
    
    arima_model.forecast(30)
   
    
    
    from statsmodels.tsa.arima.model import ARIMA
    
    model = ARIMA(data, order=(p,i,q))
    
    model_fit = model.fit()
    forecast = model_fit.predict(start=0, end=len(data)+30)
    
    plt.plot(range(len(data)), data)
    
    plt.plot(forecast)
    plt.xlabel("Lag")
    
    plt.ylabel(f"y_axis", rotation=90)
    plt.show()
    print(model_fit.summary())