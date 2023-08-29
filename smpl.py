# Copyright 2016-2023 Chong Sun (sunchong137@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/env python
'''importance sampling of the Hilbert space
'''

import numpy as np
import numpy.linalg as nl
import logger # this is my logger
import random

def smpl_hilbert(hop, ci0, T, ftlan, nblock = 100, genci = 0, nrot = 200, nw = 25, nsamp = 100, dr = 0.5, dtheta = 20.0):
    '''
       with 1 initial vector
       note that the probability and the energy is given by the same function (ftlan)
       ci0    ---- array, initial vector
       T      ---- float, temperature
       ftlan  ---- function giving the probability and energy evaluated from the initial vector
       nblock ---- the size of the block for block average
       genci  ---- int, the way to generate new ci
       nrot   ---- int, steps of rotations on ci0, only used when genci = 1
       nw     ---- number of warmup steps
       nsamp  ---- number of samples (initial vector generated)
       dr     ---- the size of the sampling box (or sphere)
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
        if cases == 1: # generate new vector by rotational FIXME gives very bad result!!!!! -_-!!!
    #        print "generating new vectors by rotation"
            v1 = v0.copy()
            for i in range(nrot):
                addr = np.random.randint(lc, size = 2)         
                theta = random.random() * dtheta # rotational angle
                vec = np.zeros(2)
                vec[0] = v1[addr[0]]
                vec[1] = v1[addr[1]]
                rotmat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
                vec = (rotmat.dot(vec)).copy()
                v1[addr[0]] = vec[0]
                v1[addr[1]] = vec[1]
            return v1/nl.norm(v1) 
 
    # Warm-up
    logger.section("Warming up ......")
    Nar = 0 # acceptance number
    tp = ftlan(hop, ci0, T, m=20)[1]
    for i in range(nw):
        ci = gen_nci(ci0, genci)
        tp_n = ftlan(hop, ci, T, m=20)[1]
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
           
    logger.section("Sampling......")
    # Sampling with block average
    Nar = 0
    Eb = [] # the array of energies per block
    e, tp = ftlan(hop, ci0, T)
    for i in range(nsamp):
        E = 0.
        for j in range(nblock):
            E += e/tp
#            print "E", e/tp
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
        E = E/(1.*nblock)
        Eb.append(E)
    Eb = np.asarray(Eb)
    E = np.mean(Eb)
    dev = np.std(Eb)
    ar =  (1.* Nar)/(1.*nblock* nsamp)
    return E, dev, ar  
