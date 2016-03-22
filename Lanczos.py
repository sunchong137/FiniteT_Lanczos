#!/usr/local/bin/python
""" This script is used to diagonalze a large sparce matrix with Lanczos method
    History: Adeline C. Sun Mar. 07 2016 first sketch
             Adeline C. Sun Mar. 21 2016 Restart Ohahaha

    chongs0419@gmail.com
"""

import numpy as np
import math
import numpy.linalg as nl
import random

N = 1000
M = 20
dens = 0.01
Nloop = 10000

def gen_mat(L = N, d = dens):
"""Generate a sparse symmetry matrix. This will not really be used... Just for
    the test of the program
"""
    n = int(L ** 2 * d)
    mat = np.empty((n, 3)) 
    for i in range(n//2):
        r = random.randint(0, L)
        c = random.randint(r, L) # c > r
        mat[i, 0], mat[i, 1] = r, c
        mat[i, 2] = random.uniform(-1., 1.)
    return mat

def vec_mat_vec(mat, vec): # <vector|matrix|vector>
    res = 0.0
    for i in range(len(mat)):
        res += vec[mat[i, 0]] * mat[i, 2] * vec[mat[i, 1]]
        res += vec[mat[i, 1]] * mat[i, 2] * vec[mat[i, 0]]
    return res

def mat_vec(mat, vec): #matrix|vector>
    res = []
    for i in range(len(mat)):
        res.append(mat[i, 2] * (vec[mat[i, 1]] + vec[mat[i, 0]]))
    res = np.asarray(res)
    return res

def Lanczos(mat, n = N, m = M):
    """ Get the tridiagonal entries
    """
    a, b = [], [] # tri diagonal vectors of H
    v0 = np.random.randn(n)
    vn = v0.copy()
    a.append(vec_mat_vec(mat, v0))
    v1, v2 = [], []
    Hv = mat_vec(mat, v0)
    v1 = Hv - a[0] * v0
    b.append(np.linalg.norm(v1))
    v1 = v1 / b[0]
    
    for i in range(m - 2):
        a.append(vec_mat_vec(mat, v1))
        v2 = mat_vec(mat, v1) - b[i - 1] * v0 - a[i] * v1
        b.append(np.linalg.norm(v2))
        v2 = v2/b[i]
        v0 = v1.copy()
        v1 = v2.copy()
    a.append(mat_vec_mat(mat, v1))
    b = np.asarray(b)
    a = np.asarray(a)
    return a, b # vn is the trial vector

def Tri_diag(a, b):
    mat = np.diag(b, -1) + np.diag(a, 0) + np.diag(b, 1) # generate the tridiag matrix
    e, w = nl.eig(mat)
    return e, w

def FT_Lanczos(H, A, Beta, n = N, m = M):
    r"""Calculate the canonical ensemble average of A with finite temperature Lanczos algorithm
    Suppose A is also a sparse matrix, usually A is just H.
    Here c is an array of <n|\phi_j>, d is an array of <\phi_j|A|n>
    """

    av_A, av_Z = 0., 0. # <A> * Z and Z
    for cnt in range(Nloop):
        a, b = [], []
        c, d = [], []
        v0 = np.random.randn(n)
        vn = v0.copy()
        Hv = mat_vec(H, v0)
        Av = mat_vec(A, v0) # I didn't mean it...I'm pure and inocent

        a.append(vec_mat_vec(H, v0))
        c.append(vn.dot(v0))
        d.append(v0.dot(Av))
        v1 = Hv - a[0] * v0
        b.append(np.linalg.norm(v1))
        v1 = v1/b[0]
        a.append(vec_mat_vec(H, v1))
        c.append(vn.dot(v1))
        d.append(v1.dot(Av))

        for i in range(m - 2):
            v2 = mat_vec(H, v1) - b[i - 1] * v0 - a[i] * v1
            b.append(np.linalg.norm(v2))
            v2 = v2/b[i]
            a.append(mat_vec_mat(H, v2))
            c.append(vn.dot(v2))
            d.append(v2.dot(Av))
            v0 = v1.copy()
            v1 = v2.copy()
        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)
        d = np.asarray(d)
        eps, phi = Tri_diag(a, b)
        eps = -Beta * np.exp(eps)
        for i in range(m):
            av_A += eps[i] * c.dot(phi[i, :]) * d.dot(phi[i, :])
            av_Z += eps[i] * c.dot(phi[i, :]) ** 2
    av_A = av_A/av_Z
    return av_A
# TODO def LT_Lanczos():

if __name__ == "__main__":
    print "Hi, This is the main function:P"
            

