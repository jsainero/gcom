# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:41:00 2020

@author: Jorge Sainero
"""

import numpy as np
from PIL import Image
import math

#Recibe una matriz de 0s y 1s y la pinta
def pintar(a,dim):
    dibujo=np.empty((dim,dim,3), dtype=np.uint8)
    for i in range(dim):
        for j in range(dim):
            if a[i,j]==1:
                #negro
                dibujo[i,j]=[0,0,0]
            else:
                #amarillo
                dibujo[i,j]=[255,233,0]
    Image.fromarray(dibujo).save("Sierpinski.png")

def sierpinski(a,dim):
    if dim != 1:
        aux = dim//3
        #rellena el interior de 0s
        a[aux:2*aux,aux:2*aux] = 0
        for i in range(0,dim,aux):
            for j in range(0,dim,aux):
                #llamada recursiva menos a la del centro
                if i != aux or j != aux:
                    sierpinski(a[i:i+aux,j:j+aux],aux)

def apartado1():
    it = 8
    dim = 3**it
    alfombra=np.ones((dim,dim))
    sierpinski(alfombra,dim)
    pintar(alfombra,dim)
    return 0;

#Recibe una alfombra (o culaquier cosa a recubrir) y 
#la dimensión de los recubridores (matrices cuadradas en este caso)
#devuelve cuantas son necesarias para recubrir la alfombra
def numRecs(alfombra,rec):
    total=0
    dim=alfombra.shape[0]
    ndim=math.ceil(dim/rec)*rec
    nalfombra=np.zeros((ndim,ndim))
    nalfombra[0:dim,0:dim]=alfombra
    for i in range(0,ndim,rec):
        for j in range(0,ndim,rec):
            if not np.array_equiv(nalfombra[i:i+rec,j:j+rec],np.zeros((rec,rec))):
                total+=1      
    return total

#Devuelve el d-volumen dados la alfombra y el diámetro de los recubridores
def volumenDdim(alfombra,rec,d):
    return numRecs(alfombra,rec)*rec**d

#Devuelve cierto si la lista es creciente
def creciente(lista):
    for i in range(len(lista)-1):
        if lista[i]>lista[i+1]:
            return False;
    return True;

def apartado2():
    it = 8
    dim = 3**it
    alfombra=np.ones((dim,dim))
    sierpinski(alfombra,dim)
    #Generamos los diámetros de los recubridores
    recs=[3**x for x in range(it,-1,-1)]
    a=1 #Extremo izquierdo (sabemos que su d-volumen es mayor que el de una línea)
    b=2 #Extremo derecho (sabemos que su d-volumen es menor que el de un cuadrado)
    #Búsqueda binaria
    for i in range(20):
        c=(a+b)/2
        volumenes=[volumenDdim(alfombra,x,c)for x in recs]
        if creciente(volumenes):
            a=c
        else:
            b=c
    print("La dimensión de la alfombra de Sierpinski es:",c)
    
apartado1()
apartado2()