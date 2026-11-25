# Analyse-num-rique-matricielle
#Question 3

import numpy as np

#Modélisation de la matrice A
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

#Décomposition LU de la matrice A
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

#Résolution du système Ly=pi
def descente(A,b):
    n=len(b)
    x=np.zeros(n,dtype=float)
    for i in range(0,n):
        s=0
        for j in range(0,i):
            s+=A[i,j]*x[j]
        x[i]=(b[i]-s)/(A[i,i])
    return x


#Résolution du système Ux=y

def remontee(A,b):
    n=len(b)
    x=np.zeros(n,dtype=float)
    for i in range(n-1,-1,-1): #on parcourt les lignes de la matrice, n-1 est la derniere ligne, on s arrete à la premiere ligne , indice 0
        s=0
        for j in range(i+1,n):
            s+=A[i,j]*x[j]
        x[i]=(b[i]-s)/(A[i,i])
    return x

#Résolution du probleme pi=Ax
n=20
tau=1.0
A=MatriceA(n,tau)
pi=np.ones(n,dtype=float)
L,U=LU1(A)
y=descente(L,pi)
x=remontee(U,y)
print ("La solution x du probleme (4) est:")
print(x)


    

def Etape1(t):
    n = len(t)
    F = []
    # Initialisation (k = 1) : f_1 = (1 / t_0)
    f1 = np.array([1.0 / t[0]], dtype=float)
    F.append(f1)
    # Boucle de récurrence pour k allant de 2 à n
    for k in range(2, n + 1):
        f_prev = F[-1]  # C'est le vecteur f_{k-1}
        delta_k = np.dot(t[1:k], f_prev) # 1. Calcul de delta_k = sum_{m=1}^{k-1} t_m * (f_{k-1})_m . t[1:k] correspond à (t_1, ..., t_{k-1})
        # Résolution du système 2x2 pour obtenir alpha_k et beta_k
        beta_k = 1.0 / (1.0 - delta_k**2)
        alpha_k = -delta_k * beta_k
        # 3. Construction de f_k = alpha_k * [S_{k-1} f_{k-1} ; 0] + beta_k * [0 ; f_{k-1}]
        v1 = np.append(f_prev[::-1], 0.0) # f_prev[::-1] réalise l'inversion d'indices S_{k-1} f_{k-1}
        v2 = np.insert(f_prev, 0, 0.0)
        f_k = alpha_k * v1 + beta_k * v2
        F.append(f_k)   
    return F

def Etape2(t, b):
    n = len(t)
    # Appel de la première étape pour récupérer (f_1, ..., f_n)
    F = Etape1(t)
    # Initialisation (k = 1) : x_1 = (b_1 / t_0)
    x = np.array([b[0] / t[0]], dtype=float)
    # Boucle de récurrence pour k allant de 2 à n
    for k in range(2, n + 1):
        f_k = F[k - 1]
        # Calcul de theta_k = b_k - sum_{j=1}^{k-1} t_{k-j} * (x_{k-1})_j
        theta_k = b[k - 1] - np.dot(t[1:k][::-1], x)
        # Mise à jour du point : x_k = [x_{k-1} ; 0] + theta_k * f_k
        x = np.append(x, 0.0) + theta_k * f_k
    return x
