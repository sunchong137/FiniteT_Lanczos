#!/usr/local/bin/python
import numpy as np
"""
Modules for calculating finite temperature properties of the system.
Adeline C. Sun Mar. 28 2016 <chongs0419@gmail.com>
"""

def Tri_diag(a1, b1):
    mat = np.diag(b1, -1) + np.diag(a1, 0) + np.diag(b1, 1)
    e, w = np.linalg.eigh(mat)
    return e, w


def ftlan_E(hop, v0, T, kB=1, m=60, Min_b=10e-5, Min_m=40, norm = np.linalg.norm):
    r"""1 step Lanczos
    Calculating the energy of the system at finite temperature.
    args:
        hop     - function to calculate $H|v\rangle$
        v0      - random initial vector
        T       - temperature
        kB      - Boltzmann const
        m       - size of the Krylov subspace
        Min_b   - min tolerance of b[i]
        Min_m   - min tolerance of m
    return:
        Energy
    """
#   def Tri_diag(a1, b1):
#       mat = np.diag(b1, -1) + np.diag(a1, 0) + np.diag(b1, 1)
#       e, w = np.linalg.eigh(mat)
#       return e, w

    N = len(v0)
    beta = 1./(T * kB)
    E, Z = 0., 0.
    a, b = [], []
    v0 = v0/norm(v0)
    Hv = hop(v0)
    a.append(v0.dot(Hv))
    v1 = Hv - a[0] * v0
    b.append(norm(v1))
    if b[0] < Min_b:
        return 0

    v1 = v1/b[0]
    Hv = hop(v1)
    a.append(v1.dot(Hv))

    for i in range(1, m - 1):
        v2 = Hv - b[i - 1] * v0 - a[i] * v1
        b.append(norm(v2))
        v2 = v2/b[i]
        if abs(b[i]) < Min_b:
            b.pop()
            break

        Hv = hop(v2)
        a.append(v2.dot(Hv))
        v0 = v1.copy()
        v1 = v2.copy()
    
    a = np.asarray(a)
    b = np.asarray(b)

    eps, phi = Tri_diag(a, b)
    exp_eps = np.exp(-beta * eps)
    for i in range(len(eps)):
        E += exp_eps[i] * eps[i] * phi[0, i]**2
        Z += exp_eps[i] * phi[0, i]**2

    E = E/Z
    return E

# not finish!                
#def LT_lanczos_E(hop, v0, T, kB=1, m=60, nsamp=100, Min_b=10e-5, Min_m=40, norm=np.linalg.norm):
#   r"""
#   Calculate the energy of the system at low temperature
#   """
#   def Tri_diag(a1, b1):
#       mat = np.diag(b1, -1) + np.diag(a1, 0) + np.diag(b1, 1)
#       e, w = np.linalg.eigh(mat)
#       return e, w

def FT_Lanczos_1rdms(qud, hop, v0, T, kB=1, m=60, Min_b=10e-5, Min_m=30,norm=np.linalg.norm):
    r'''1 step lanczos
    return the 1st order reduced density matrix
    at finite temperature.
    args:
        qud    - function for getting the matrix repr
                 of the RDM of given two vectors
        hop    - function to get $H|v\rangle$
        v0     - initial vector (normalized)
        T      - temperature
        kB     - Boltzmann const
        m      - size of the Krylov subspace
        Min_b  - min tolerance of b[i] 
        Min_m  - min tolerance of m
    return:
        RDMs of spin a and b
    '''
    rdma, rdmb = qud(v0, v0)*0. #so we don't need norb
    Z = 0.
    a, b = [], []
    krylov = []
    v0 = v0/norm(v0)
    krylov.append(v0)
    Hv = hop(v0)
    a.append(v0.dot(Hv))
    v1=Hv - a[0]*v0
    b.append(norm(v1))
    if b[0] < Minb:
        return 0
    v1 = v1/b[0]
    Hv = hop(v1)
    a.append(v1.dot(Hv))
    krylov.append(v1)
    for i in range(1, m-1):
        v2 = Hv - b[i-1]*v0-a[i]*v1
        b.append(norm(v2))
        if abs(b[i])<Min_b:
            if i < Min_m:
                return 0
            b.pop()
            break
        v2 = v2/b[i]
        krylov.append(v2)
        Hv = hop(v2)
        a.append(v2.dot(Hv))
        v0 = v1.copy()
        v1 = v2.copy()
    
    a, b = np.asarray(a), np.asarray(b)
    krylov = np.asarray(krylov)
    eps, phi = Tri_diag(a, b)
    coef = np.exp(-beta*eps/2.)*phi[0, :]
    eps = np.exp(-beta*eps)
    for i in range(len(eps)):
        for j in range(len(eps)):
            for cnt1 in range(m):
                for cnt2 in range(m):
                    tmpa, tmpb = coef[i]*coef[j]*phi[cnt1,i]*phi[cnt2,j]*qud(krylov[cnt1, :], krylov[cnt2,:])
                    rdma += tmpa
                    rdmb += tmpb

        Z += eps[i]*phi[0, i]**2.

    rdma = rdma/Z
    rdmb = rdmb/Z
    return rdma, rdmb

            
#TODO 
# use symmetry to reduce the comutational expense
# make the initial guess a function and pass it to the loop
# make loops
# the whole RDM
