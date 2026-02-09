# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 22:33:24 2026

@author: John Martin

"""
"""_________________________________________________________________________"""

"""____________________IMPORTING THE REQUIRED PACKAGES______________________"""


import numpy as np
import matplotlib.pyplot as plt
from src.acf.acf_cpp_backend import acf_cpp_backend_calc
from src.tools.confidence_interval import confidence_interval_plot
from typing import Literal

"""_________________________________________________________________________"""

"""_____ACF_PY FUNCTION IS WHOLLY PYTHON ACF FUNCTION HAS A C++ BACKEND_____"""

"""____________________________ACF_PY FUNCTIONS_____________________________"""

def acf_numerator(sample: [list, np.array, tuple], lag: int,
                  sample_mean: float, gamma_0: float) -> float:
    
    gamma_s= np.sum((sample[lag:] - sample_mean)*(sample[:-lag] - sample_mean)) if lag > 0 else gamma_0
    return gamma_s


def correlation_sum(sample: [list, np.array, tuple], lags: int) -> [list, np.array, tuple]:
    
    correlations = []
    sample_mean = np.mean(sample)
    gamma_0 = np.sum((sample-sample_mean)**2)
    
    for lag in range(lags):
        gamma_s = acf_numerator(sample, lag, sample_mean, gamma_0)
        correlations.append(gamma_s/gamma_0)
    
    return correlations


def acf_py(sample, lags: int = 50, return_acf: bool = False,
           plot_acf: bool = True,
           confidence_bounds: float = 0.95) -> [list, np.array, tuple]:
    lags = lags+1
    
    if lags-2 >= len(sample):
        print ("Number of lags exceeds data length")
        return
    
    acf = correlation_sum(sample, lags)
    
    '''plot the acf results'''
    if plot_acf == True:
        acf_plotting(acf, sample, lags, confidence_bounds)
        
    if return_acf == True:
        return acf
    
"""_________________________________________________________________________"""
    
"""_____________________ACF WITH C++ BACKEND FUNCTION_______________________"""

def acf(sample, lags: int = 50, avx: [Literal["avx", "avx2", "auto"], None] = "auto",
        return_acf: bool = False, plot_acf: bool = True, confidence_bounds: float = 0.95, 
        ) -> [list, np.array, tuple]:
    lags = lags + 1
    
    if lags - 2 >= len(sample):
        print ("Number of lags exceeds data length")
        return

    acf = acf_cpp_backend_calc(sample, lags, str(avx))

    if plot_acf == True:
        acf_plotting(acf, sample, lags, confidence_bounds)
        
    if return_acf == True:
        return acf

"""_________________________________________________________________________"""

"""__________________________ACF PLOTTING FUNCTION__________________________"""

def acf_plotting(acf: [list, np.array, tuple], sample: [list, np.array, tuple],
                 lags: int, confidence_bounds: float ) -> None:
        
        for num , val in enumerate(acf):
            plt.arrow(num, 0, 0, val, head_width = 0.5,head_length=0.03, width=0.05, color='b', ec='b', length_includes_head=True)
        confidence_interval_plot(acf, sample, lags, confidence_bounds)
        plt.title("Autocorrelation")
        plt.ylabel("ACF")
        plt.xlabel("lag")
        plt.hlines(0, len(acf)+1, -2, color = 'b', linewidth = 1)
        
        plt.ylim(-1.1,1.1)



    
    