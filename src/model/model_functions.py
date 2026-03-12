# -*- coding: utf-8 -*-
"""
Created on Thu March  12 13:07:24 2026

@author: John Martin

"""
"""_________________________________________________________________________"""

"""____________________IMPORTING THE REQUIRED PACKAGES______________________"""


import numpy as np
from statistics import stdev
from itertools import product
from typing import Tuple
#import matplotlib.pyplot as plt
#from src.acf.acf_cpp_backend import acf_cpp_backend_calc
#from src.tools.confidence_interval import confidence_interval_plot
#from typing import Literal

def MA_builder(x:float) -> float:
    return 2 + 1 * np.random.normal(0,0.1) #- 1.2 * np.random.normal(0.5,1.4)

def MA_constructor() -> np.array:
    MAmodel = []
    for i in range(100):
        MAmodel.append(MA_builder(MAmodel))
    return np.array(MAmodel)



# def likelihood_py(theta_hat, x, y, sigma:float = 1):
#     x = np.array(x)
#     y = np.array(y)
#     theta_hat = np.array(theta_hat)
#     y_bin = y
#     for i in theta_hat:
#         y_bin -= i*x
#     exponential_exponent =  (y_bin)**2 / (2 * sigma**2)
#     exponential = np.exp(-1*exponential_exponent)
#     front = 1/(np.sqrt(2 * np.pi * sigma**2))
#     return np.sum(np.log(front * exponential))

# print(likelihood_py([1,1.3,2.7,-4], [1,2,3], [3.7,2.1,4.5] ))

data = MA_constructor()

def MA_likelihood_py(theta: [list, np.array], y: [list, np.array]) -> float :
    y = np.array(y)
    
    y_minus_mean_y = y - np.mean(y)
    
    sigma = np.std(y_minus_mean_y)
    
    x = np.array(np.random.normal(0,sigma, size=y.size))
    
    theta_hat = np.array(theta)
    
    y_minus_theta_x = y_minus_mean_y
    
    for i in theta_hat:
        y_minus_theta_x -= i*x
    
    gaussian_probability = (1/np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(y_minus_theta_x)**2 / (2 * sigma**2))
    
    return np.sum(np.log(gaussian_probability))

#print(MA_likelihood_py([0.1,0.1,-0.2], data))


def MA_fit(q: int, series_data: [list, np.array], min_theta_q: float = -1,
           max_theta_q: float = 1, step_theta_q: float =0.5) -> Tuple[np.array, float, float] :
    prob_dict = {}
    
    q_range = np.arange(min_theta_q, max_theta_q+step_theta_q , step_theta_q)
    
    q_values = np.tile(q_range, (q,1))
    
    for combination in product(*q_values):
        prob_dict[combination] = MA_likelihood_py(combination, series_data)
    
    max_key = max(prob_dict, key=prob_dict.get)
    
    return np.array(max_key), np.mean(series_data), np.std(series_data)


def model_func(x, parameters: [list, np.array], mean : float, stdev: float, MA_fitting = None):
    if MA_fitting==None:
        
print(MA_fit(3,data))
    
    

    
