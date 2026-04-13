# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:59:19 2026

@author: John Martin
"""
import numpy as np


'''__________________________________________Stationality functions__________________________________________________________________________________________'''

def stationality_phi(phi):
    
    if len(phi)== 1:
    
        return True if -1<=phi[0]<=1 else False
    
    else:
        parameter = []
        
        for i in phi:
            parameter.append(-i)
        
        parameter.append(1)
        
        roots = np.roots(parameter)
       
        return all([abs(r)>1 for r in roots])
 
    
 
    
 
def stationality_theta(theta):
    
    if len(theta)== 1:
    
        return True if -1<=theta[0]<=1 else False
    
    else:
        parameter = []
        
        for i in theta:
            parameter.append(-i)
        
        parameter.append(1)
        
        roots = np.roots(parameter)
       
        return all([abs(r)>1 for r in roots])


'''_________________________________________________MA constructor_________________________________________________________________________________'''




'''_____________________________________________________AR constructor______________________________________________________________________________'''


    
'''_____________________________________ARMA constructor_____________________________________________________________________________________________'''


