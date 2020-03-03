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
f = nc.netcdf_file("air.2019.nc", 'r')

print(f.history)
print(f.dimensions)
print(f.variables)
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
print(lats)

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
    
for max_lat in range(len(lats)-1,-1,-1):
    if lats[max_lat]==-20:
        min_lat=max_lat
    if lats[max_lat]==20:
        break

for max_lon in range(len(lons)):
    if lons[max_lon]==30:
        min_lon=max_lon
    if lons[max_lon]==50:
        break
    

print(max_lat,min_lat)
print(min_lon,max_lon)
subair = air[t,:,max_lat:min_lat,min_lon:max_lon]

print(air[t,10,20,0])
























