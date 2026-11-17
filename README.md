# Analyse-num-rique-matricielle
#Question 3

import numpy as np

def MatriceA(n,tau):
    #Initialisation d'une matrice nulle de taille n x n
    A=np.zeros((n,n))
    #Parcours de chaque colonne j de la matrice de 0 à n-1, soit n colonnes
    for j in range(n):
        #Calcule du coefficient pour chaque ligne i de la colonne j
        for i in range(n):
            #V(d)=1/(1+tau**2) avec tau=d/r0
            A[i,j]= 1.0/(1.0+(tau * np.abs(i-j))**2)
    return A


MatriceA(3,1)
def LU1(A):
    A=np.array(A, dtype=float)
    n=A.shape[0]
    L=np.eye(n)
    U=A.copy()
    for k in range(n-1):
        if U[k, k] == 0:
            raise ValueError(f"Pivot nul rencontré à l'indice ({k}, {k}). La décomposition LU sans pivotement échoue.")
        for i in range(k + 1, n):
            m=U[i, k]/U[k, k]
            L[i, k]=m
            U[i, k:]-=m*U[k, k:]        
    return L, U
MatriceA(20,1)
A=MatriceA(20,1)
LU1(A)
