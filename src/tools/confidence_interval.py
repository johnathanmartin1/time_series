# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 22:39:41 2026

@author: John Martin
"""
"""_________________________________________________________________________"""

"""________________IMPORTING THE REQUIRED PACKAGES__________________________"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""_________________________________________________________________________"""

"""_____________________ADJUSTABLE CONFIDENCE BOUNDS________________________"""


def confidence_bounds_z_value(confidence: float) -> float:
    alpha = 1 - confidence
    z = norm.ppf(1-alpha/2)
    return z
    
#Bartlett Confidence interval with adjustable accuracy
def confidence_interval(acf:[list, np.array, tuple], 
                        sample: [list, np.array, tuple],
                        lags: int, confidence_bound: float ) -> [list, np.array, tuple]:
    n = len(sample)
    stderr_a = [1/np.sqrt(n)]
    stderr_b = [np.sqrt((1+2*np.sum(np.array(acf[1:k])**2))/n) for k in range(1, lags)]
    stderr = np.array(stderr_a + stderr_b)
    pos_con_interval = (confidence_bounds_z_value(confidence_bound)) * stderr
    neg_con_interval = [-1*i for i in pos_con_interval]
    return pos_con_interval, neg_con_interval

"""_________________________________________________________________________"""

"""______________CONFIDENCE BOUNDS PLOTTING TOOL____________________________"""

def confidence_interval_plot(acf_pacf_data: [list, np.array, tuple],
                             sample_data: [list, np.array, tuple], lags: int,
                             confidence_bounds: float ) -> None:
    con = confidence_interval(acf_pacf_data, sample_data, lags, confidence_bounds)
    x = [i+0.5 for i in range(len(acf_pacf_data))]
    plt.fill_between(x, con[0], con[1], color='g', alpha=0.2, ec=None)
    
"""_________________________________________________________________________"""