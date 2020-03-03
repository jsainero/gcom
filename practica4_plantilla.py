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

#workpath = "C:/Users/Robert/Documents/NCEP"
#os.getcwd()
#os.chdir(workpath)
#files = os.listdir(workpath)


#f = nc.netcdf_file(workpath + "/" + files[0], 'r')
f = nc.netcdf_file("hgt.2019.nc", 'r')


###############################################
#cambiar los air por hgt en las variables y todo
##############################################

print(f.history)
print(f.dimensions)
print(f.variables)
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
level = f.variables['level'][:].copy() #los valores de p
lats = f.variables['lat'][:].copy() #los valores de y
lons = f.variables['lon'][:].copy() #los valores de x
air = f.variables['hgt'][:].copy() #los valores de cada día
air_units = f.variables['hgt'].units
#tem=f.variables['air']
#tem_scale_factor=tem.scale_factor.copy()
#tem_add_offset=tem.add_offset.copy()
print(air.shape)
f.close()
print(len(lats))

"""
Ejemplo de evolución temporal de un elemento de aire
"""
plt.plot(time, air[:, 1, 1, 1], c='r')
plt.show()
print(len(lats))
#time_idx = 237  # some random day in 2012
# Python and the renalaysis are slightly off in time so this fixes that problem
# offset = dt.timedelta(hours=0)
# List of all times in the file as datetime objects
dt_time = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) #- offset\
           for t in time]
np.min(dt_time)
np.max(dt_time)

"""
Distribución espacial de la temperatura en el nivel de 500hPa, para el primer día
"""

plt.contour(lons, lats, air[1,1,:,:])
plt.show()

PRESION = 500
for i in range(len(level)):
    if level[i]==PRESION:
        break

air2 = air[:,i,:,:].reshape(len(time),len(lats)*len(lons))
#air3 = air2.reshape(len(time),len(lats),len(lons))
n_components=4

X = air2
Y = air2.transpose()
pca = PCA(n_components=n_components)

pca.fit(X)
print(pca.explained_variance_ratio_)
out = pca.singular_values_

pca.fit(Y)
print(pca.explained_variance_ratio_)
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

"""
apartado 2
"""

f = nc.netcdf_file("air.2020.nc", 'r')
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
level = f.variables['level'][:].copy() #los valores de p
lats = f.variables['lat'][:].copy() #los valores de y
lons = f.variables['lon'][:].copy() #los valores de x
air = f.variables['air'][:].copy() #los valores de cada día
air_units = f.variables['air'].units
print(air.shape)

f.close()


for t in range(len(time)):
    if dt.date(1800, 1, 1) + dt.timedelta(hours=time[t]) == dt.date(2020,1,20):
        break
    
#tienen que ir de -20 a 20 
double_lons=np.concatenate((lons, lons), axis=None)
#print(double_lons)
min_lon=-1
for max_lon in range(len(double_lons)):
    if double_lons[max_lon]==340:
        min_lon=max_lon
    if double_lons[max_lon]==20 and not min_lon==-1:
        break

#tiene que ir de 30 a 50 pero está ordenado al reves
for max_lat in range(len(lats)):
    if lats[max_lat]==50:
        min_lat=max_lat
    if lats[max_lat]==30:
        break
    

print(min_lat,max_lat)
print(min_lon,max_lon)


d0 = air[t,:,min_lat:max_lat,min_lon:max_lon]

IND_500=5
IND_1000=0

#cargar los datos de 2019
def dist_euc(d0,d):
    dist=0
    for i in range(max_lat,min_lat+1):
        for j in range(min_lon,max_lon+1):
            dist+=0.5*(d0[i,j%len(lons),IND_500]-d[i,j%len(lons),IND_500])**2+0.5*(d0[i,j%len(lons),IND_1000]-d[i,j%len(lons),IND_1000])**2
    return np.sqrt(dist)
























