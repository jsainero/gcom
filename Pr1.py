# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:14:07 2020

@author: usu321
"""
#poner errores

import numpy as np
import random as rand
import matplotlib.pyplot as plt

PINTAR_GRAFICA = True
EPSILON = 1e-6;

def logistica(x, r):
    return r*x*(1-x)

def fn(x0,f,n, r):
    x=x0
    for j in range(n):
        x=f(x, r)
    return x
    

def afinar(v0,f, r):
    for i in range(v0.size):
        v0[i]=fn(v0[i],f,100, r);
    return v0;        


def atractor(x, r):
    k = 25;
    n = 500
    orb =  np.zeros(n)
    for i in range(n):
        orb[i] = fn(x,logistica,i, r)
   
    abscisas = np.linspace(0,n-1,n)
    if PINTAR_GRAFICA:
        plt.plot(abscisas,orb)
    ult=orb[-1*np.arange(k,0,-1)]
    #print (ult)
    
    periodo = -1
    for i in range(1,k,1):
        if abs(ult[k - 1] - ult[k - i - 1]) < EPSILON:
            periodo = i;
            break;
    if periodo == -1:
        print ("No se ha encontrado un periodo")
    else:
        print("Periodo: "+str(periodo))
        v0 = orb[-1*np.arange(periodo,0,-1)];
        v0 = afinar(v0,logistica, r);
    return periodo
    

def apartado1():
    global PINTAR_GRAFICA
    PINTAR_GRAFICA = True;
    r1 = rand.uniform(3.0001 ,3.4999);
    r2 = rand.uniform(3.0001 ,3.4999);
    x01 = rand.random();
    x02 = rand.random();
    atractor(x01, r1);
    atractor(x02, r2);

def apartado2():
    global PINTAR_GRAFICA
    PINTAR_GRAFICA = False;
    x0 = rand.random();
    a = 3; 
    b = 4;
    cardinalV0 = -1;
    #Iteramos hasta encontar un r tal que V0 tenga 8 elementos
    while cardinalV0 != 8:
        r = (a+b)/2;
        cardinalV0 = atractor(x0, r);
        if cardinalV0 > 8 or cardinalV0 == -1:
            b = r;
        elif cardinalV0 < 8:
            a = r;
    print(a,atractor(x0,a))
    print(b,atractor(x0,b))
    #Buscamos los extremos del intervalo de los valores de r tal que el cardinal de la órbita es 8
    a1 = a;
    b1 = r;
    a2 = r;
    b2 = b;
    for i in range(20):
        r1 = (a1 + b1)/2;
        cardinalV01 = atractor(x0,r1);
        r2 = (a2 + b2)/2;
        cardinalV02 = atractor(x0,r2);
        if cardinalV01 < 8:
            a1 = r1;
        else:
            b1 = r1;
            
        if cardinalV02 > 8 or cardinalV02 == -1:
            b2 = r2;
        else:
            a2 = r2;
    print ([b1,a2])

apartado2()


        
    
    
    
    
    
    
    
    
    
    
    
    