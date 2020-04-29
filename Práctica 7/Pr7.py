# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:58:33 2020

@author: Robert
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from skimage import io, color


"""
Ejemplo para el apartado 1.

Modifica la figura 3D y/o cambia el color
https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
"""
def animate2(t):
    M = np.array([[np.cos(3*np.pi*t),-np.sin(3*np.pi*t),0],[np.sin(3*np.pi*t),np.cos(3*np.pi*t),0],[0,0,1]])
    
    v=np.array([d,d,0])*t
    
    ax = plt.axes(xlim=(0,100), ylim=(0,100), projection='3d')
    #ax.view_init(60, 30)

    X,Y,Z = transf1D(x-bx, y-by, z, M=M, v=v)
    X=X.reshape(xshape)
    Y=Y.reshape(yshape)
    Z=Z.reshape(zshape)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,alpha=0.5,
                    cmap='viridis', edgecolor='none')
    return ax,

def init2():
    return animate2(0),

fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contour(X, Y, Z, 16, extend3d=True,cmap = plt.cm.get_cmap('viridis'))
ax.clabel(cset, fontsize=9, inline=1)
plt.show()

x=np.asarray(X).reshape(-1)
y=np.asarray(Y).reshape(-1)
z=np.asarray(Z).reshape(-1)
xshape=X.shape
yshape=Y.shape
zshape=Z.shape
d = 100
bx=np.mean(x)
by=np.mean(y)

animate2(np.arange(0.1, 1,0.1)[5])
plt.show()


fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate2, frames=np.arange(0,1,0.025), init_func=init2,
                              interval=20)

ani.save("p7a.gif", fps = 10)  


def animate(t):
    M = np.array([[np.cos(3*np.pi*t),-np.sin(3*np.pi*t),0],[np.sin(3*np.pi*t),np.cos(3*np.pi*t),0],[0,0,1]])
    
    v=np.array([d,d,0])*t
    
    ax = plt.axes(xlim=(0,400), ylim=(0,400), projection='3d')
    #ax.view_init(60, 30)

    XYZ = transf1D(x0-bx, y0-by, z0, M=M, v=v)
    col = plt.get_cmap("viridis")(np.array(0.1+XYZ[2]))
    ax.scatter(XYZ[0],XYZ[1],c=col,s=0.1,animated=True)
    return ax,

def init():
    return animate(0),

"""
Transformación para el segundo apartado

NOTA: Para el primer aparado es necesario adaptar la función o crear otra similar
pero teniendo en cuenta más dimensiones
"""

def transf1D(x,y,z,M, v=np.array([0,0,0])):
    xt = np.empty(len(x))
    yt = np.empty(len(x))
    zt = np.empty(len(x))
    for i in range(len(x)):
        q = np.array([x[i],y[i],z[i]])
        xt[i], yt[i], zt[i] = np.matmul(M, q) + v
    return xt, yt, zt



"""
Segundo apartado casi regalado

Imagen del árbol
"""


img = io.imread('arbol.png')
dimensions = color.guess_spatial_dimensions(img)
print(dimensions)
io.show()
#io.imsave('arbol2.png',img)

#https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
fig = plt.figure(figsize=(5,5))
p = plt.contourf(img[:,:,0],cmap = plt.cm.get_cmap('viridis'), levels=np.arange(0,240,2))
plt.axis('off')
#fig.colorbar(p)

xyz = img.shape

x = np.arange(0,xyz[0],1)
y = np.arange(0,xyz[1],1)
xx,yy = np.meshgrid(x, y)
xx = np.asarray(xx).reshape(-1)
yy = np.asarray(yy).reshape(-1)
z  = img[:,:,0]
zz = np.asarray(z).reshape(-1)


"""
Consideraremos sólo los elementos con zz < 240 

Por curiosidad, comparamos el resultado con contourf y scatter!
"""
#Variables de estado coordenadas
x0 = xx[zz<240]
y0 = yy[zz<240]
z0 = zz[zz<240]/256.
#Variable de estado: color
col = plt.get_cmap("viridis")(np.array(0.1+z0))

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 2, 1)
plt.contourf(x,y,z,cmap = plt.cm.get_cmap('viridis'), levels=np.arange(0,240,2))
ax = fig.add_subplot(1, 2, 2)
plt.scatter(x0,y0,c=col,s=0.1)
plt.show()

def diametro(x,y):
    d=0
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            aux=(x[i]-x[j])**2+(y[i]-y[j])**2
            if (aux>d):
                d=aux
    return np.sqrt(d)

#d = diametro(x0,y0)
bx=np.mean(x0)
by=np.mean(y0)



def init():
    return animate(0),

animate(np.arange(0.1, 1,0.1)[5])
plt.show()


fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,1,0.025), init_func=init,
                              interval=20)

ani.save("p7b.gif", fps = 10)  








