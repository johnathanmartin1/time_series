# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:28:07 2026

@author: John Martin
"""
"""select tests to be run with Triue and False switches"""
console_output = True
acf_functions_test_switch = False
pacf_functions_test_switch = True


"""_________________________________________________________________________"""

"""__________________IMPORTING THE REQUIRED PACKAGES________________________"""

import numpy as np 
import matplotlib.pyplot as plt
#import statsmodels.api as sm 

"""_________________________________________________________________________"""

"""_______________TOY MODELS FOR TESTING FUNCTION OUTPUTS___________________"""

def MA_builder(x:float) -> float:
    return 2 + 1 * np.random.normal(2,0.1) #- 1.2 * np.random.normal(0.5,1.4)

def MA_constructor() -> np.array:
    MAmodel = []
    for i in range(100):
        MAmodel.append(MA_builder(MAmodel))
    return np.array(MAmodel)

def AR_constructor() -> np.array:
        ARmodel=[1,1,1]
        for i in range(0,200):
            ARmodel.append(ARmodel[-1]*0.25+ ARmodel[-2]*0.25-
                           ARmodel[-3]*0.25 +np.random.normal(0.5,0.75))
        return np.array(ARmodel) 
    
"""_________________________________________________________________________"""
    
"""__________________________ACF FUNCTION TESTS_____________________________"""

if acf_functions_test_switch == True:
    
    from time_series import acf_py, acf
    
    def acf_functions_test() -> None:
     
        model = MA_constructor()
        
        """testing the python acf_py function"""
        #sm.graphics.tsa.plot_acf(model,lags=50, fft=False)
        acf_py(model, 50, return_acf=False)
        plt.show()
        
        if console_output == True:
            print("acf_py function result length (should be 1 more than requested lags) =",
                  len(acf_py(model,50,return_acf=True, plot_acf=False)),"\n")
            
            print("acf_py function results =",acf_py(model,50,return_acf=True, plot_acf=False))
            
            print("\n\n\n")
        
        """testing the cpp backend acf function"""
        #sm.graphics.tsa.plot_acf(model,lags=50, fft=False)
        acf(model, 50, return_acf=False)
        plt.show()
        
        if console_output == True:
            print("acf function result length (should be 1 more than requested lags) =",
                  len(acf(model,50,return_acf=True, plot_acf=False)), "\n")
            
            print("acf function results =",acf(model,50,return_acf=True, plot_acf=False))
            
            print("\n\n\n")
    
    acf_functions_test()

"""_________________________________________________________________________"""

"""_________________________PACF FUNCTION TESTS_____________________________""" 

if pacf_functions_test_switch == True:
    
    from time_series import pacf_py, pacf
    
    def pacf_functions_test() -> None:
        model = AR_constructor()
        
        """testing the python pacf_py function"""
        
        #sm.graphics.tsa.plot_pacf(model,lags=50)
        pacf_py(model, 50, plot_pacf=True, confidence_bounds=0.95)
        plt.show()
        
        if console_output == True:
            print("pacf_py function results length (should be 1 more than requested lags) =",
                  len(pacf_py(model, 50,plot_pacf=False, return_pacf=True)), "\n")
            
            print("pacf function results =",pacf_py(model,50,return_pacf=True, plot_pacf=False))
            
            print("\n\n\n")
            
        
        """testing the python pacf function"""
        
        #sm.graphics.tsa.plot_pacf(model,lags=50)
        pacf(model, 50, plot_pacf=True, confidence_bounds=0.95)
        plt.show()
        
        if console_output == True:
            print("pacf function results length (should be 1 more than requested lags) =",
                  len(pacf(model, 50,plot_pacf=False, return_pacf=True)), "\n")
            
            print("pacf function results =",pacf(model,50,return_pacf=True, plot_pacf=False))
            
            print("\n\n\n")
    
        
    pacf_functions_test()

"""_________________________________________________________________________"""