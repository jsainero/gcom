"""
Práctica 2
"""

#import os
import numpy as np
import pandas as pd
import math
from itertools import accumulate as acc
import matplotlib.pyplot as plt
from collections import Counter

#### Carpeta donde se encuentran los archivos ####
#ubica = "C:/Users/Python"

#### Vamos al directorio de trabajo####
#os.getcwd()
#os.chdir(ubica)
#files = os.listdir(ubica)

with open('auxiliar_en_pract2.txt', 'r') as file:
      en = file.read()
     
with open('auxiliar_es_pract2.txt', 'r') as file:
      es = file.read()

#### Pasamos todas las letras a minúsculas
en = en.lower()
es = es.lower()

#### Contamos cuantas letras hay en cada texto
from collections import Counter
tab_en = Counter(en)
tab_es = Counter(es)

##### Transformamos en formato array de los carácteres (states) y su frecuencia
##### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_en_states = np.array(list(tab_en))
tab_en_weights = np.array(list(tab_en.values()))
tab_en_probab = tab_en_weights/float(np.sum(tab_en_weights))
distr_en = pd.DataFrame({'states': tab_en_states, 'probab': tab_en_probab})
distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index=np.arange(0,len(tab_en_states))

tab_es_states = np.array(list(tab_es))
tab_es_weights = np.array(list(tab_es.values()))
tab_es_probab = tab_es_weights/float(np.sum(tab_es_weights))
distr_es = pd.DataFrame({'states': tab_es_states, 'probab': tab_es_probab })
distr_es = distr_es.sort_values(by='probab', ascending=True)
distr_es.index=np.arange(0,len(tab_es_states))

##### Para obtener una rama, fusionamos los dos states con menor frecuencia
distr = distr_en
''.join(distr['states'][[0,1]])

### Es decir:
states = np.array(distr['states'])
probab = np.array(distr['probab'])
state_new = np.array([''.join(states[[0,1]])])   #Ojo con: state_new.ndim
probab_new = np.array([np.sum(probab[[0,1]])])   #Ojo con: probab_new.ndim
codigo = np.array([{states[0]: 0, states[1]: 1}])
states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
distr = pd.DataFrame({'states': states, 'probab': probab, })
distr = distr.sort_values(by='probab', ascending=True)
distr.index=np.arange(0,len(states))

#Creamos un diccionario
branch = {'distr':distr, 'codigo':codigo}

## Ahora definimos una función que haga exáctamente lo mismo
def huffman_branch(distr):
    states = np.array(distr['states'])
    probab = np.array(distr['probab'])
    state_new = np.array([''.join(states[[0,1]])])
    probab_new = np.array([np.sum(probab[[0,1]])])
    codigo = np.array([{states[0]: 0, states[1]: 1}])
    states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
    probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
    distr = pd.DataFrame({'states': states, 'probab': probab, })
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index=np.arange(0,len(states))
    branch = {'distr':distr, 'codigo':codigo}
    return(branch) 

def huffman_tree(distr):
    tree = np.array([])
    while len(distr) > 1:
        branch = huffman_branch(distr)
        distr = branch['distr']
        code = np.array([branch['codigo']])
        tree = np.concatenate((tree, code), axis=None)
    return(tree)
 
distr = distr_en 
tree = huffman_tree(distr)
tree[0].items()
tree[0].values()

#Buscar cada estado dentro de cada uno de los dos items
list(tree[0].items())[0][0] ## Esto proporciona un '0'
list(tree[0].items())[1][0] ## Esto proporciona un '1'

"""
A partir de aqu´ı, c´odigo escrito por:
- Jorge Sainero Valle
- Lucas de Torre Barrio
"""

#Extrae el código binario de Huffman de cada carácter y lo devuelve en un diccionario {carácter : código}
def extract_code():
    global tree
    d = dict()
    #Empezamos por el final porque es donde está la raíz
    for i in range(tree.size-1,-1,-1):
        elem=tree[i]
        h1=list(elem.keys())[0]
        h2=list(elem.keys())[1]

        for c in h1:
            if c in d:
                d[c]+='0'
            else:
                d[c]='0'

        for c in h2:
            if c in d:
                d[c]+='1'
            else:
                d[c]='1'
    return d

#Calcula la longitud del código binario de Huffman 
def longitudMedia(d):
    ac=0
    for i,k in distr.iterrows():
        ac+=len(d[k['states']])*k['probab']
    return ac

#Calcula la entropía de una variable aleatoria con la distribución de probabilidades distr['probab']
def entropia():
    h=0
    for p in distr['probab']:
        h-=p*math.log(p,2)
    return h
        
    
        
def apartado1():
    print("APARTADO UNO:\n")
    global tree
    global distr
    distr = distr_en
    tree = huffman_tree(distr)
    d=extract_code()
    print("Una pequeña muestra del diccionario con la codificación de Senglish:",dict(Counter(d).most_common(7)),'\n')
    print ("La longitud media de Senglish es: "+str(longitudMedia(d)))       
    h_en=entropia()
    print("La entropía de Senglish es: ",h_en)
    print("Vemos que se cumple el Teorema de Shannon ya que",h_en,"<=",str(longitudMedia(d)),"<",h_en+1,"\n")
    
    
    distr = distr_es
    tree = huffman_tree(distr)
    d=extract_code()
    print("Una pequeña muestra del diccionario con la codificación de Sspanish:",dict(Counter(d).most_common(7)),'\n')
    print ("La longitud media de Sspanish es: "+str(longitudMedia(d)))       
    h_es=entropia()
    print("La entropía de Sspanish es: ",h_es)
    print("Vemos que se cumple el Teorema de Shannon ya que",h_es,"<=",str(longitudMedia(d)),"<",h_es+1,"\n\n")
 
#Codifica la palabra pal según el diccionario d
def codificar(pal, d):
    binario=""
    for l in pal:
        binario += d[l]
    return binario
    
def apartado2():
    print("APARTADO DOS:\n")
    global tree
    global distr
    distr = distr_en
    tree = huffman_tree(distr)    
    d_en=extract_code()
    palabra = "fractal"
    pal_bin =codificar(palabra,d_en)
    print("El codigo binario de la palabra fractal en lengua inglesa es:",pal_bin,"y su longitud es:",len(pal_bin))
    #La longitud en binario usual es ceil(log2(cantidadDeCracteres)) por cada letra
    print("En binario usual sería de longitud:", len(palabra)*math.ceil(math.log(len(d_en),2)),"\n\n")
    distr = distr_es
    tree = huffman_tree(distr)    
    d_es=extract_code()
    pal_bin =codificar(palabra,d_es)
    print("El codigo binario de la palabra fractal en lengua española es:",pal_bin,"y su longitud es:",len(pal_bin))
    #La longitud en binario usual es ceil(log2(cantidadDeCracteres)) por cada letra
    print("En binario usual sería de longitud:", len(palabra)*math.ceil(math.log(len(d_es),2)),"\n\n")

#Decodifica la palabra pal según el diccionario d
def decodificar(pal,d):
    aux=''
    decode=''
    for i in pal:
        aux+=i
        if aux in d:
            decode+=d[aux]
            aux=''
    return decode
    
def apartado3():
    print("APARTADO TRES:\n")
    global tree
    global distr
    distr = distr_en
    tree = huffman_tree(distr)    
    d_en=extract_code()
    #Invertimos el diccionario para poder decodificar
    d_en_inv=dict(zip(list(d_en.values()),list(d_en.keys())))
    pal_bin='1010100001111011111100'
    palabra=decodificar(pal_bin,d_en_inv)
    print("La palabra cuyo código binario es:",pal_bin,"en inglés es:",palabra,end='\n\n\n')

#Calcula el índice de Gini de una variable aleatoria con la distribución de probabilidades distr['probab']  
def gini():
    aux=0
    #ya está ordenado así que no hace falta ordenarlo
    accu=list(acc(distr['probab']))
    plt.plot(np.linspace(0,1,len(accu)),accu)
    plt.plot(np.linspace(0,1,len(accu)),np.linspace(0,1,len(accu)))
    plt.show()
    for i in range(1,len(accu)):
        aux+=(accu[i]+accu[i-1])/len(accu)
    return 1-aux

#Calcula la diversidad 2D de Hill de una variable aleatoria con la distribución de probabilidades distr['probab'] 
def diver2hill():
    aux=0
    for p in distr['probab']:
        aux+=p*p
    return 1/aux

    
def apartado4():
    print("APARTADO CUATRO:\n")
    global distr
    distr = distr_en   
    print("El índice de Gini de Senglish es: ",gini())
    print("La diversidad 2D de Hill de Senglish es: ",diver2hill())
    distr = distr_es
    print("El índice de Gini de Sspanish es: ",gini())
    print("La diversidad 2D de Hill de Sspanish es: ",diver2hill())

apartado1()
apartado2() 
apartado3()
apartado4()  
    
    
    
    
    
    
    
    
    