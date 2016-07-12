#!/usr/local/bin/python
import numpy as np
"""
Modules for calculating finite temperature properties of the system.
Adeline C. Sun Mar. 28 2016 <chongs0419@gmail.com>
Adeline C. Sun Apr. 11 Made some corrections. Added the RDM function
"""

def Tri_diag(a1, b1):
    mat = np.diag(b1, -1) + np.diag(a1, 0) + np.diag(b1, 1)
    e, w = np.linalg.eigh(mat)
    return e, w


def ftlan_E1c(hop, v0, T, m=50, Min_b=10e-10, Min_m=5, kB=1, norm = np.linalg.norm):
    r"""1 cycle Lanczos... you need to generate a random number first and then
        pass it into this function, iteration is built outside.

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
        if succeed: Energy
        if b[0]=0 : 0
    """
#   def Tri_diag(a1, b1):
#       mat = np.diag(b1, -1) + np.diag(a1, 0) + np.diag(b1, 1)
#       e, w = np.linalg.eigh(mat)
#       return e, w

    beta = 1./(T * kB)
    E = 0.
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
        if abs(b[i]) < Min_b:
            b.pop()
            break

        v2 = v2/b[i]
        Hv = hop(v2)
        a.append(v2.dot(Hv))
        v0 = v1.copy()
        v1 = v2.copy()
    
    a = np.asarray(a)
    b = np.asarray(b)

    eps, phi = Tri_diag(a, b)
    l = len(eps)
#    Eo = eps[0]
#    eps = eps-Eo
    exp_eps = np.exp(-beta * eps)
    E = np.sum(exp_eps * eps * phi[0, :]**2.)
    Z = np.sum(exp_eps * phi[0, :]**2.)
#    for i in range(len(eps)):
#        E += exp_eps[i] * eps[i] * phi[0, i]**2

#    E = E + Eo
#   de = eps[:, np.newaxis] - eps
#   for i in range(l):
#       E += eps[i] * phi[0, i]**2./np.sum(np.exp(-beta*de[:l, i])*(phi[0, :l]**2.))
    return E, Z

def ftlan_E(hop, vecgen, T, m=60, nsamp=30, Min_b=10e-5, Min_m=5, kB=1):
    r'''Multi-cycle Lanczos. can iterate inside, yeah! Inheritade from ftlan_E1c
    new args:
        vecgen - function to generate an initial vector
        nsamp  - number of sample initial vectors
    return:
        Energy
    '''
    beta = 1./(kB*T)
    cnt = nsamp
    E = 0.
    Z = 0.
    while cnt > 0:
        v0 = vecgen()
        etmp, ztmp = ftlan_E1c(hop, v0, T, m, Min_b, Min_m, kB)
#       print cnt, etmp
        if etmp==0:
            continue
        E += etmp
        Z += ztmp
        cnt -= 1

    e = E/Z
    return e

#def ltlan_E1c(hop, r0, T, m=60, Min_b=10e-5, Min_m=5,kB=1, norm=np.linalg.norm):
#    beta = 1./(T*kB)
#    E, Z = 0., 0.,
#    a, b = [], []
#    krylov = []
#    hvspace = []
#    v0 = (r0/norm(r0)).copy()
#    Hv = hop(v0)
#    hvspace.append(Hv.copy())
#    a.append(v0.dot(Hv))
#    v1 = Hv - a[0]*v0
#    b.append(norm(v1))
#    if b[0] < Min_b:
#        return 0
#    v1 = v1/b[0]
#    krylov.append(v0.copy())
#    krylov.append(v1.copy())
#    Hv = hop(v1)
#    hvspace.append(Hv.copy())
#    a.append(v1.dot(Hv))
#
#    for i in range(1, m-1):
#        v2 = Hv - b[i - 1] * v0 - a[i] * v1
#        b.append(norm(v2))
#        if abs(b[i]) < Min_b:
#            b.pop()
#            break
#        v2 = v2/b[i]
#        krylov.append(v2.copy())
#        Hv = hop(v2)
#        hvspace.append(Hv.copy())
#        a.append(v2.dot(Hv))
#        v0 = v1.copy()
#        v1 = v2.copy()
#
#    a = np.asarray(a)
#    b = np.asarray(b)
#    krylov = np.asarray(krylov)
#    hvspace = np.asarray(hvspace)
#
#    eps, phi = Tri_diag(a, b)
#    l = len(eps)
#    Eo = eps[0]
#    eps = (eps - Eo).copy()
#    exp_eps = np.exp(-beta * eps)
#    exp_half = np.exp(-beta * eps/2.)
#    for i in range(l):
#        Z += exp_eps[i] * phi[0, i]**2
#    for cnt1 in range(l):
#        for cnt2 in range(l):
#            inner = krylov[cnt1, :].dot(hvspace[cnt2, :])    
#            for i in range(l):
#                for j in range(l):
#                    E += phi[0, i]*phi[0, j] * exp_half[i]*exp_half[j]*inner*phi[cnt1, i] * phi[cnt2, j]
#
#    E = E/Z + Eo
#    return E
 
#def ltlan_E(hop, vecgen, T, m=60, nsamp=30, Min_b=10e-5, Min_m=5, kB=1):
#    E = 0.
#    cnt = nsamp
#    while cnt > 0:
#        v0 = vecgen()
#        etmp = ltlan_E1c(hop, v0, T, m, Min_b, Min_m, kB)
#        if etmp==0:
#            continue
#        E += etmp
#        cnt -= 1
#
#    E = E/float(nsamp)
#    return E

def ftlan_rdm1s1c(qud, hop, v0, T, norb, m=60, Min_b=10e-10, Min_m=5, kB=1, norm=np.linalg.norm):
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
#    rdma, rdmb = qud(v0, v0)*0. #so we don't need norb
    beta = 1./(kB*T)
    rdma, rdmb = np.zeros((norb, norb)), np.zeros((norb, norb)) 
    Z = 0.
    a, b = [], []
    krylov = []
    v0 = v0/norm(v0)
    krylov.append(v0)
    Hv = hop(v0)
    a.append(v0.dot(Hv))
    v1=Hv - a[0]*v0
    b.append(norm(v1))
    if b[0] < Min_b:
        return 0, 0
    v1 = v1/b[0]
    Hv = hop(v1)
    a.append(v1.dot(Hv))
    krylov.append(v1)
    for i in range(1, int(m-1)):
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
    eps = eps-eps[0]
    coef = np.exp(-beta*eps/2.)*phi[0, :]
    eps = np.exp(-beta*eps)
    l = len(eps)
    Z = np.sum(eps*phi[0, :]**2.)
    for cnt1 in range(l):
        tmpa, tmpb = qud(krylov[cnt1, :], krylov[cnt1,:])
        for i in range(l):
            for j in range(l):
                rdma += (coef[i]*coef[j]*phi[cnt1,i]*phi[cnt1,j])*tmpa
                rdmb += (coef[i]*coef[j]*phi[cnt1,i]*phi[cnt1,j])*tmpb
        for cnt2 in range(cnt1+1, l):
            tmpa, tmpb = qud(krylov[cnt1, :], krylov[cnt2,:])
            for i in range(l):
                for j in range(l):
                    rdma += 2.*(coef[i]*coef[j]*phi[cnt1,i]*phi[cnt2,j])*tmpa
                    rdmb += 2.*(coef[i]*coef[j]*phi[cnt1,i]*phi[cnt2,j])*tmpb

#   for i in range(l):
#       for j in range(l):
#           for cnt1 in range(l):
#               for cnt2 in range(l):
#                   tmpa, tmpb = qud(krylov[cnt1, :], krylov[cnt2,:])
#                   tmpa = (coef[i]*coef[j]*phi[cnt1,i]*phi[cnt2,j]/Z)*tmpa
#                   tmpb = (coef[i]*coef[j]*phi[cnt1,i]*phi[cnt2,j]/Z)*tmpb
#                   rdma += tmpa
#                   rdmb += tmpb

    return rdma, rdmb, Z

def ftlan_rdm1s(qud, hop, vecgen, T, norb, m=50, nsamp=20, Min_b=10e-10, Min_m=30, kB=1):
#    v0 = vecgen()
#    rdma, rdmb = qud(v0, v0)*0. # can use np.zeros((norb, norb))
    rdma, rdmb = np.zeros((norb, norb)), np.zeros((norb, norb))
    cnt = nsamp
    Z = 0.
    while cnt > 0:
        v0 = vecgen()
        tmpa, tmpb, tmpz=ftlan_rdm1s1c(qud, hop, v0, T, norb, m, Min_b, Min_m, kB)
        if isinstance(tmpa, int):
            continue
        rdma += tmpa
        rdmb += tmpb
        Z += tmpz
        cnt -= 1

    rdma = rdma/Z
    rdmb = rdmb/Z
    return rdma, rdmb


#def ht_rdm1s1c(qud, hop, v0, T, norb, m=60, Min_b=10e-10, Min_m=5, kB=1, norm=np.linalg.norm):
#   beta = 1./(kB*T)
#   rdma, rdmb = np.zeros((norb, norb)), np.zeros((norb, norb))
#   Z = 0.
#   a, b = [], []
#   D1, D2 = [], []
#   v0 = v0/norm(v0)
#   vr = v0.copy()
#   d1, d2 = qud(v0, vr)
#   D1.append(d1)
#   D2.append(d2)
#   Hv = hop(v0)
#   a.append(v0.dot(Hv))
#   v1=Hv - a[0]*v0
#   b.append(norm(v1))
#   if b[0] < Min_b:
#       return 0, 0
#   v1 = v1/b[0]
#   d1, d2 = qud(v1, vr)
#   D1.append(d1)
#   D2.append(d2)
#   Hv = hop(v1)
#   a.append(v1.dot(Hv))
#   for i in range(1, int(m-1)):
#       v2 = Hv - b[i-1]*v0-a[i]*v1
#       b.append(norm(v2))
#       if abs(b[i])<Min_b:
#           if i < Min_m:
#               return 0
#           b.pop()
#           break
#       v2 = v2/b[i]
#       d1, d2 = qud(v2, vr)
#       D1.append(d1)
#       D2.append(d2)
#       Hv = hop(v2)
#       a.append(v2.dot(Hv))
#       v0 = v1.copy()
#       v1 = v2.copy()
#   
#   a, b = np.asarray(a), np.asarray(b)
#   D1 = np.asarray(D1)
#   D2 = np.asarray(D2)
#   eps, phi = Tri_diag(a, b)
#   eps = eps-eps[0]
#   coef = np.exp(-beta*eps)*phi[0, :]
#   eps = np.exp(-beta*eps)
#   l = len(eps)
#   for i in range(l):
#       Z += eps[i]*phi[0, i]**2.
#   coef = coef/Z
#   for j in range(l):
#       tmpa, tmpb = np.zeros((norb, norb)), np.zeros((norb, norb))
#       for k in range(l):
#           tmpa += phi[k, j]*D1[k]
#           tmpb += phi[k, j]*D2[k]
#       rdma += coef[j]* tmpa
#       rdmb += coef[j]* tmpb

#   rdma = rdma
#   rdmb = rdmb
#   return rdma, rdmb

#def ht_rdm1s(qud, hop, vecgen, T, norb, m=50, nsamp=20, Min_b=10e-10, Min_m=30, kB=1):
#    v0 = vecgen()
#    rdma, rdmb = qud(v0, v0)*0. # can use np.zeros((norb, norb))
#   rdma, rdmb = np.zeros((norb, norb)), np.zeros((norb, norb))
#   cnt = nsamp
#   while cnt > 0:
#       v0 = vecgen()
#       tmpa, tmpb=ht_rdm1s1c(qud, hop, v0, T, norb, m, Min_b, Min_m, kB)
#       if isinstance(tmpa, int):
#           continue
#       rdma += tmpa
#       rdmb += tmpb
#       cnt -= 1

#   rdma = rdma/float(nsamp)
#   rdmb = rdmb/float(nsamp)
#   return rdma, rdmb
