# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 17:59:30 2026

@author: John Martin
"""

"""_________________________________________________________________________"""

"""_____________________IMPORTING THE REQUIRED PACKAGES_____________________"""
from src.acf.acf_function import acf
import numpy as np
import matplotlib.pyplot as plt
from src.tools.confidence_interval import confidence_interval_plot
from src.pacf.pacf_cpp_backend import pcorr_128

"""_________________________________________________________________________"""

"""_______PACF_PY IS WHOLLY PYTHON PACF FUNCTION HAS A C++ BACKEND__________"""

"""___________________________PACF_PY FUNCTION______________________________"""

def pacf_py(sample: [list, np.array] , lags: int = 20, return_pacf: bool = False,
            plot_pacf: bool = True, confidence_bounds: float = 0.95) -> [None, list, np.array]:
    lags += 1
    pacf: list = [1]
    auto_corr = np.array(acf(sample, lags, return_acf=True, plot_acf=False))
    for lag in range(1, lags):
        pacfn_numerator = auto_corr[lag] - np.sum([pacf[k]*auto_corr[lag-k] for k in range(1,lag)])
        pacfn_denominator = 1 - np.sum([pacf[k]*auto_corr[k] for k in range(1,lag)])
        pacf.append(pacfn_numerator/pacfn_denominator)
        
    if plot_pacf == True:
        pacf_plotting(pacf, sample, lags, confidence_bounds) 
        
    if return_pacf == True:
        return pacf


def pacf(sample: [list, np.array] , lags: int = 20, return_pacf: bool = False,
            plot_pacf: bool = True, confidence_bounds: float = 0.95) -> [None, list, np.array]:
    lags += 1
    auto_corr = acf(sample, lags, return_acf=True, plot_acf=False)
        
    pacf = pcorr_128(auto_corr, lags)
    
        
    if plot_pacf == True:
        pacf_plotting(pacf, sample, lags, confidence_bounds) 
        
        
    if return_pacf == True:
        return pacf
        
"""_________________________________________________________________________"""

"""_________________________PACF PLOTTING FUNCTION__________________________"""

def pacf_plotting(pacf: [list, np.array, tuple], sample: [list, np.array, tuple],
                  lags: int, confidence_bounds):
        
        for num , val in enumerate(pacf):
            plt.arrow(num, 0, 0, val, head_width = 0.5,head_length=0.03, width=0.05, color='b', ec='b', length_includes_head=True)
        confidence_interval_plot(pacf, sample, lags, confidence_bounds)
        plt.title("Partial - Autocorrelation")
        plt.ylabel("PACF")
        plt.xlabel("lag")
        plt.hlines(0, len(pacf)+1, -2, color = 'b', linewidth = 1)
        
        plt.ylim(-1.1,1.1)



    
   