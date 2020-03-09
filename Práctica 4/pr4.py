# -*- coding: utf-8 -*-
"""
Referencias:
    
    Fuente primaria del reanálisis
    https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis2.pressure.html
    
    Altura geopotencial en niveles de presión
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=59&tid=81620&vid=1498
    
    Temperatura en niveles de presión:
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=59&tid=81620&vid=1497
"""
#import os
import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf as nc
from sklearn.decomposition import PCA
import math

"""
Apartado 1
"""

def apartado1():
    
    #Cargamos los datos de altura geopotencial del 2019
    f = nc.netcdf_file("hgt.2019.nc", 'r')    
    #print(f.history)
    #print(f.dimensions)
    #print(f.variables)
    time = f.variables['time'][:].copy()
    #time_bnds = f.variables['time_bnds'][:].copy()
    #time_units = f.variables['time'].units
    level = f.variables['level'][:].copy() #los valores de p
    lats = f.variables['lat'][:].copy() #los valores de y
    lons = f.variables['lon'][:].copy() #los valores de x
    hgt = f.variables['hgt'][:].copy() #los valores de cada día
    #hgt_units = f.variables['hgt'].units
    #alt=f.variables['hgt']
    #alt_scale_factor=tem.scale_factor.copy()
    #alt_add_offset=tem.add_offset.copy()
    #print(hgt.shape)
    f.close()
    
    """
    Ejemplo de evolución temporal de un elemento de hgte
    
    plt.plot(time, hgt[:, 1, 1, 1], c='r')
    plt.show()
    """
    #time_idx = 237  # some random day in 2012
    # Python and the renalaysis are slightly off in time so this fixes that problem
    # offset = dt.timedelta(hours=0)
    # List of all times in the file as datetime objects
    dt_time = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) #- offset\
               for t in time]
    np.min(dt_time)
    np.max(dt_time)
    
    PRESION = 500
    for i in range(len(level)):
        if level[i]==PRESION:
            break
    p500=i
    
    plt.title("Distribución espacial de la altura geopotencial en el nivel de 500hPa, para el primer día")
    plt.contour(lons, lats, hgt[0,p500,:,:])
    plt.show()
    
   
    
    hgt2 = hgt[:,p500,:,:].reshape(len(time),len(lats)*len(lons))
    #hgt3 = hgt2.reshape(len(time),len(lats),len(lons))
    n_components=4
    
    X = hgt2
    Y = hgt2.transpose()
    pca = PCA(n_components=n_components)
    
    pca.fit(X) #(creo) aqui intenta explicar la posición en función de la hgt
    print(pca.explained_variance_ratio_.cumsum())
    out = pca.singular_values_
    
    pca.fit(Y) #(creo) aqui al revés la hgt en función de la posición, 
    #esto es más lógico y por eso se obtienen mejores resultados
    print(pca.explained_variance_ratio_.cumsum())
    out = pca.singular_values_
    
    
    State_pca = pca.fit_transform(X)
    
    #Ejercicio de la práctica
    Element_pca = pca.fit_transform(Y)
    Element_pca = Element_pca.transpose(1,0).reshape(n_components,len(lats),len(lons))
    print(Element_pca[0,0,0])
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, n_components+1):
        ax = fig.add_subplot(2, 2, i)
        ax.text(0.5, 90, 'PCA-'+str(i),
               fontsize=18, ha='center')
        plt.contour(lons, lats, Element_pca[i-1,:,:])
    plt.show()
    print("\n\n\n\n")
    
    
    

"""
apartado 2
"""
IND_500=5
IND_1000=0

#Calculala distancia euclídea entre dos días en un subconjunto del sistema
def dist_euc(d0,d,min_lat,max_lat,min_lon,max_lon,lons):
    dist=0
    for i in range(min_lat+1,max_lat):
        for j in range(min_lon+1,max_lon):
            dist+=0.5*(d0[IND_500,i,j%lons]-d[IND_500,i,j%lons])**2+0.5*(d0[IND_1000,i,j%lons]-d[IND_1000,i,j%lons])**2
    return math.sqrt(dist)


def apartado2():
    print("Apartado 2:\n")
    
    #Cargamos los datos de altura geopotencial del 2020
    f = nc.netcdf_file("hgt.2020.nc", 'r')
    time = f.variables['time'][:].copy()
    lats = f.variables['lat'][:].copy() #los valores de y
    lons = f.variables['lon'][:].copy() #los valores de x
    hgt = f.variables['hgt'][:].copy() #los valores de cada día
    f.close()
    
    #Calculamos el índice correspondiente al día 2020/01/20
    for t in range(len(time)):
        if dt.date(1800, 1, 1) + dt.timedelta(hours=time[t]) == dt.date(2020,1,20):
            break
        
    double_lons=np.concatenate((lons, lons), axis=None)
    min_lon=-1
    for max_lon in range(len(double_lons)):
        if double_lons[max_lon]==340:
            min_lon=max_lon
        if double_lons[max_lon]==20 and not min_lon==-1:
            break
    
    for max_lat in range(len(lats)):
        if lats[max_lat]==50:
            min_lat=max_lat
        if lats[max_lat]==30:
            break
    
    
    #cargamos los datos de altura geopotencial del día 2020/01/20
    d0h = hgt[t,:,:,:]
    
    
    #cargamos los datos de temperatura del 2020
    f = nc.netcdf_file("air.2020.nc", 'r')
    air = f.variables['air'][:].copy() #los valores de cada día
    tem_scale_factor=f.variables['air'].scale_factor.copy()
    tem_add_offset=f.variables['air'].add_offset.copy()
    f.close()
       
    #cargamos los datos de altura geopotencial del día 2020/01/20
    d0a = air[t,0,:,:]*tem_scale_factor+tem_add_offset
    
    
    #cargamos los datos de altura geopotencial del 2019    
    f = nc.netcdf_file("hgt.2019.nc", 'r')
    lons = f.variables['lon'][:].copy() #los valores de x
    hgt = f.variables['hgt'][:].copy() #los valores de cada día
    f.close()
    
    
    
    #Calculamos la distancia de cada día de nuestro subsistema al día 2020/01/20
    distancias=[[dist_euc(d0h,hgt[i,:,:,:],min_lat,max_lat,min_lon,max_lon,len(lons)),i]for i in range(hgt.shape[0])]
    distancias.sort()
    
    take=[j for i,j in distancias[0:4]]
    print("Los 4 días más análogos son:",take,"\n")
    
    #cargamos los datos de temperatura de 2019    
    f = nc.netcdf_file("air.2019.nc", 'r')
    air = f.variables['air'][:].copy() #los valores de cada día
    tem_scale_factor=f.variables['air'].scale_factor.copy()
    tem_add_offset=f.variables['air'].add_offset.copy()
    f.close()
    
    #Transformamos los datos para que estén en grados Kelvin
    air=air*tem_scale_factor+tem_add_offset
    
    #Calculamos la media de los 4 días más análogos a 2020/01/20
    mediaDias=np.mean(air[take,0,:,:],axis=0)
    
    print("El error absoluto medio de la temperatura prevista para el día a0 es",np.mean(abs(mediaDias-d0a)))


apartado1()
apartado2()
