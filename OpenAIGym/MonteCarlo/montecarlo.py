# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 21:37:53 2017

@author: ahsan
"""

import numpy as np

sample = 1000000
ok = 0

def funct(x,y):
    val = x*x + y*y
    val = np.sqrt(val)
    if val <= 1.0:
        return 1
    else :
        return 0

cnt = 0


for s in range(sample):
    x = np.random.rand()
    y = np.random.rand()
    cnt += funct(x,y)
        
    
print(cnt)
pii = 4*( (cnt*1.0)/(sample*1.0) )
print(pii)
print(sample)
print("differenece : ")
print( np.abs(np.pi-pii)/np.pi * 100)
