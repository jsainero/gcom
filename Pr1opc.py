# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:41:00 2020

@author: jsain y ldeto
"""

import numpy as np
from colorama import Back

def pintar(a,dim):
    for i in range(dim):
        for j in range(dim):
            if a[i,j]==1:
                print(Back.BLACK + "  ", end = '')
            else:
                print(Back.YELLOW + "  ", end = '')
        print()

def sierpinski(a,dim):
    if dim != 1:
        aux = dim//3
        a[aux:2*aux,aux:2*aux] = 0
        for i in range(0,dim,aux):
            for j in range(0,dim,aux):
                if i != aux or j != aux:
                    sierpinski(a[i:i+aux,j:j+aux],aux)

def apartado1():
    it = 3
    dim = 3**it
    alfombra=np.ones((dim,dim))
    sierpinski(alfombra,dim)
    pintar(alfombra,dim)
    return 0;

def apartado2():
    return 0;

apartado1()
