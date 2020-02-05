# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:41:00 2020

@author: jsain y ldeto
"""

import numpy as np
import random as rand
import matplotlib.pyplot as plt

def apartado1():
    it=2
    dim = 3**it
    alfombra=np.ones((dim,dim))
    #for i in range(it):
    alfombra[3:5][3:5]=0
    print (alfombra)
    return 0;

def apartado2():
    return 0;
apartado1()