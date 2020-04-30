# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:58:33 2020
@author: Jorge Sainero y Lucas de Torre
"""

#from mpl_toolkits import mplot3d

import os
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d

os.getcwd()


u = np.linspace(0, np.pi, 25)
v = np.linspace(0, 2 * np.pi, 50)

#Cambiamos de coordenadas polares a cartesianas
x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

#Definimos una curva en la superficie de la esfera
t2 = np.linspace(0.001, 1, 200)
x2 = abs(t2) * np.sin(107 * t2/2)
y2 = abs(t2) * np.cos(107 * t2/2)
z2 = np.sqrt(1-x2**2-y2**2)

z0 = -1


    
def proj2(x,z,t,z0=-1,alpha=1):
    z0 = z*0+z0
    eps = 1e-16
    x_trans = x/(np.tan(np.arctan(t*abs(-z-1))*np.pi/4+(1-t)*np.pi/4)+eps)
    return(x_trans)
    #Nótese que añadimos un épsilon para evitar dividir entre 0!!
    
    

    
from matplotlib import animation
    #from mpl_toolkits.mplot3d.axes3d import Axes3D
    
def animate(t):
    xt = proj2(x,z,t)
    yt = proj2(y,z,t)
    zt = -1*t + z*(1-t)
    x2t = proj2(x2,z2,t)
    y2t = proj2(y2,z2,t)
    z2t = -1*t + z2*(1-t)
    
    ax = plt.axes(projection='3d')
    ax.set_zlim3d(-1,1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1,alpha=0.5,
                    cmap='viridis', edgecolor='none')
    ax.plot(x2t,y2t, z2t, '-b',c="gray")
    return ax,

def init():
    return animate(0),


def solucion():
   
    global x,y,z,t2,x2,y2,z2,z0
    
    t = 0.1
    
    
    xt = proj2(x,z,t)
    yt = proj2(y,z,t)
    zt = -1*t + z*(1-t)
    x2t = proj2(x2,z2,t)
    y2t = proj2(y2,z2,t)
    z2t = -1*t + z2*(1-t)
    
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')
    
    
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1,alpha=0.5,
                    cmap='viridis', edgecolor='none')
    ax.plot(x2t,y2t, z2t, '-b',c="gray")
    
   #plt.show()
    plt.close(fig) 
    
    """
    HACEMOS LA ANIMACIÓN
    """
    
    
    
    #animate(np.arange(0, 1,0.1)[1])
   # plt.show()
    
    
    fig = plt.figure(figsize=(12,12))
    ani = animation.FuncAnimation(fig, animate, np.arange(0,1,0.05), init_func=init,
                                  interval=20)
    ani.save("animaciontan.gif", fps = 5) 



solucion()
