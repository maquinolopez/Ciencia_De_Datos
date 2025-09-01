#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:55:35 2024

@author: jac
"""

### Python es un esqueleto basico al cual se la agergan
### 'modulos' de manera muy eficiente

from numpy import exp, log, array, zeros, ones, linspace
from scipy.stats import uniform, gamma, norm
from matplotlib.pylab import subplots

#from pandas import ... 

def Prueba( m, B):
    """Python pasa los parametros de funciones por referencia """
    for i in range(m):
        tmp =  B[i,i]**2
        print(tmp)
        B[i,i] = tmp #se modifica B

def Prueba2( m, B):
    """Python pasa los parametros de funciones por referencia
       Copiar datos para no usar modificar ref.
    """
    B = B.copy() 
    for i in range(m):
        tmp =  B[i,i]**2
        print(tmp)
        B[i,i] = tmp
    return B



if __name__ == "__main__":
    
    
    x = [ 1, 2, 3]  #list, las listas son mutables
    x[0] # el primer elemento es el cero (!), como en C/C++
    
    b = [ 1, exp, log, x] # las lista son de cualquier objeto
    print( exp(8))
    ## Que va a dar esto?
    print( b[3][1], b[1](8) )
    
    c = [ 0, 1, [exp, log, [3000,4000]]]
    
    y = (3,4) # tuple, no es mutable

    # Comentario de una linea
    ### Se inicializa una matriz de numpy
    A = zeros((4,4), dtype=int)
        
    """Comentario multilinea:
        range es un iterable """
    for i in range(4):
        for j in range(4):
            A[i,j] = (i+1)*10 + (j+1)
    print(A)

    X = A # No se copia A
    X[0,0] = 7
    print(A)
    
    ### Es mejor hacerlo explicito y poner
    X = A.copy()
    X[0,0] = 8
    print(A)
    
    A = uniform.rvs(size = (4,4))

    Prueba(4, A)
    print(A)
    Prueba2( 4, A)
    print(A)

    B = uniform.rvs(size = (4,4))
    print( A*B) #multiplicacion por elemento
    print( A @ B) #multiplicacion de matrices    
    
    print([1,2,3,4] * B)  
    B
    
    
    # Comentario una linea
    # Un ejemplo de graficacion
    fig, ax =subplots(nrows=1, ncols=2)

    x = linspace( -0.5, 20, num=100)
    a=3
    ax[0].plot( x/a, log(x/a), '-')
    ax[0].set_ylabel(r"$log\left( \frac{x}{a} \right) $")
    ax[0].set_xlabel(r"$\frac{x}{a}$")

    ### Definimos la Gamma( 3, 5), simulamos de ella
    ### y la graficamos junto con su pdf
    ga = gamma( 3, scale=5)
    sim = ga.rvs(size=10000)
    ax[1].hist(sim, density=True)
    t = linspace( 0, 80, num=500)
    ax[1].plot( t, ga.pdf(t), '-')

     

