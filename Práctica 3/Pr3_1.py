# -*- coding: utf-8 -*-
"""
Plantilla 1 de la práctica 3

Referencia: 
    https://scikit-learn.org/stable/modules/clustering.html
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

# #############################################################################
# Aquí tenemos definido el sistema X de 1000 elementos de dos estados
# construido a partir de una muestra aleatoria entorno a unos centros:
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)
#Si quisieramos estandarizar los valores del sistema, haríamos:
#from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X)  

plt.plot(X[:,0],X[:,1],'ro', markersize=1)
plt.show()


def grafica(labels,n_clusters):
    # Representamos el resultado con un plot
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.figure(figsize=(8,4))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)
    
    plt.title('Fixed number of KMeans clusters: %d' % n_clusters)
    plt.show()

def kMeans(k):
    global X
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    if k==1:
       silhouette = -1
    else:
        silhouette = metrics.silhouette_score(X, labels)
    print("Los centros de los",k,"clusters son:",kmeans.cluster_centers_)
    print("El coeficiente de Silhouette para nuestro sistema de 1000 elementos distribuidos en",k,"clusters es: %0.3f" % silhouette)
    grafica(labels,k)
    return silhouette

def bestSilhouette():
    sils=np.zeros(15)
    maxi=-1
    for i in range(1,16):
        sils[i-1]=kMeans(i)
        if sils[i-1]>maxi:
            maxi=sils[i-1]
            j=i
    plt.title('Valor de Silhouette en función del número de clusters')
    plt.plot(np.linspace(1,15,15),sils)
    plt.show()
    print("El valor óptimo de Silhouette es %0.3f" % maxi, "y se obtiene con", j, "clusters")
bestSilhouette()