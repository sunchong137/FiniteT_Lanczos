#!/usr/bin/env python
'''importance sampling of the Hilbert space
   Chong Sun 2016 Jun. 10
'''

import numpy as np
import numpy.linalg as nl
import random

def smpl_hilbert(hop, ci0, T, ftlan, genci = 0, nrot = 10, nw = 25, nsamp = 100, dr = 0.5, dtheta = 0.3):
    '''
       with 1 initial vector
       note that the probability and the energy is given by the same function (ftlan)
       ci0   ---- array, initial vector
       T     ---- float, temperature
       ftlan ---- function giving the probability and energy evaluated from the initial vector
       genci ---- int, the way to generate new ci
       nrot  ---- int, steps of rotations on ci0, only used when genci = 1
       nw    ---- number of warmup steps
       nsamp ---- number of samples (initial vector generated)
       dr    ---- the size of the sampling box (or sphere)
    '''
    beta = 1./T  # kB = 1.
    # generate the starting vector ----NOTE can also be several starting vectors
    ci0 = ci0.reshape(-1).copy()
    lc = len(ci0)
   
    def gen_nci(v0, cases):
        if cases == 0: # generate new vector by displacement
            disp = np.random.randn(lc) * dr
            v1 = v0 + disp
            return v1/nl.norm(v1)
        if cases == 1: # generate new vector by rotational
            v1 = v0.copy()
            for i in range(nrot):
                addr = np.random.randint(lc, size = 2)         
                theta = random.random() * dtheta # rotational angle
                vec = np.zeros(2)
                vec[0] = v1[addr[0]]
                vec[1] = v1[addr[1]]
                rotmat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
                vec = rotmat.dot(vec)
                v1[addr[0]] = vec[0]
                v1[addr[1]] = vec[1]
            return v1/nl.norm(v1) 
 
    # Warm-up
    Nar = 0 # acceptance number
    tp = ftlan(hop, ci0, T)[1]
    for i in range(nw):
        ci = gen_nci(ci0, genci)
        tp_n = ftlan(hop, ci, T)[1]
        acc = tp_n/tp 
        if acc >= 1:
            ci0 = ci.copy()
            Nar += 1
        else:
            tmp = random.random()
            if tmp <= acc:
                ci0 = ci.copy()
                tp = tp_n
                Nar += 1
            else:
                continue
            
    # Sampling
    Nar = 0
    E = 0.
    e, tp = ftlan(hop, ci0, T)
    for i in range(nsamp):
        E += e/tp
        ci = gen_nci(ci0, genci)
        e_n, tp_n = ftlan(hop, ci, T)
        acc = tp_n/tp
        if acc >= 1:
            ci0 = ci.copy()
            e = e_n
            tp = tp_n
            Nar += 1
        else:
            tmp = random.random()
            if tmp <= acc:
                ci0 = ci.copy()
                tp = tp_n
                e = e_n
                Nar += 1
            else:
                continue

    E = E/(1.*nsamp)
    ar =  (1.* Nar)/(1.* nsamp)
    print ar
    return E    
