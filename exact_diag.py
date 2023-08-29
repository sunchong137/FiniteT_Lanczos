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


#!/usr/local/bin/python
'''
test the results of FT using exact diagonalization. (for small systems lol)
'''


import numpy as np
from numpy import linalg as nl
from functools import reduce
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf.fci import direct_spin1
from pyscf.fci import cistring
import sys

def exdiagH(h1e, g2e, norb, nelec, writefile=True):
    '''
        exactly diagonalize the hamiltonian.
    '''
    h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    naa = cistring.num_strings(norb, neleca)
    nbb = cistring.num_strings(norb, nelecb)
    ndim = naa*nbb
    eyebas = np.eye(ndim, ndim)
    def hop(c):
        hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    Hmat = []
    for i in range(ndim):
        hc = hop(eyebas[i].copy())
        Hmat.append(hc)

    Hmat = np.asarray(Hmat)
#    Hmat = Hmat.T.copy()
    ew, ev = nl.eigh(Hmat.T)
    if writefile:
        np.savetxt("cards/eignE.dat", ew, fmt="%10.10f")
        np.savetxt("cards/eignV.dat", ev, fmt="%10.10f")
    return ew, ev

def fted_E(T, h1e, g2e, norb, nelec, readfile=False, kB=1., Tmin=10e-4):
    '''get the energy at temperature T using ED.
    '''
    if readfile:
        ew = np.loadtxt("cards/eignE.dat")
    else:
        ew, ev = exdiagH(h1e, g2e, norb, nelec)

    if T < Tmin:
        return ew[0]
    beta = 1./(kB * T)
    Z = np.sum(np.exp(-beta*ew))
    E = np.sum(ew*np.exp(-beta*ew))/Z
    return E

def gen_rdm1s(h1e, g2e, norb, nelec, readfile=False):
    if readfile:
        ev = np.loadtxt("cards/eignV.dat")
    else:
        ew, ev = exdiagH(h1e, g2e, norb, nelec)
#   rdma, rdmb = [], []
    ndim = len(ev)
    for i in range(ndim):
        tmpa, tmpb = direct_spin1.make_rdm1s(ev[:, i].copy(), norb, nelec)
#        tmpa, tmpb = direct_spin1.trans_rdm1s(ev[:, i], ev[:, i], norb, nelec)
        np.savetxt("cards/rdma/rdma%d"%i, tmpa)
        np.savetxt("cards/rdmb/rdmb%d"%i, tmpb)
#       rdma.append(tmpa)
#       rdmb.append(tmpb)
#   rdma = np.asarray(rdma)
#   rdmb = np.asarray(rdmb)
#   np.savetxt("cards/rdma.txt", rdma)
#   np.savetxt("cards/rdmb.txt", rdmb)

def fted_rdm1s(T, h1e, g2e, norb, nelec, readfile=False, kB=1., Tmin=10e-4):
    '''get the rdma and rdmb at T using ED
    '''
    if readfile:
        ew = np.loadtxt("cards/eignE.dat")
    else:
        ew, ev = exdiagH(h1e, g2e, norb, nelec)

#    dmas = np.loadtxt("cards/rdma.txt")
#    dmbs = np.loadtxt("cards/rdmb.txt")
    if T < Tmin:
        rdma = np.loadtxt("cards/rdma/rdma0")
        rdmb = np.loadtxt("cards/rdmb/rdmb0")
        return rdma, rdmb

    beta = 1./(kB*T)
    Z = np.sum(np.exp(-beta*ew))
    P = np.exp(-beta*ew)/Z
    ndim = len(ew)
    rdma = np.zeros((norb, norb))
    rdmb = np.zeros((norb, norb))

    for i in range(ndim):
        tmpa = np.loadtxt("cards/rdma/rdma%d"%i)
        tmpb = np.loadtxt("cards/rdmb/rdmb%d"%i)
        rdma += P[i]*tmpa
        rdmb += P[i]*tmpb
    return rdma, rdmb

if __name__ == "__main__":
    # system to be tested
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.basis = 'sto-3g'
    mol.build()
    
    m = scf.RHF(mol)
    m.kernel()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron - 2
    h1e = reduce(np.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)
   
#    ew, ev = exdiagH(h1e, eri, norb, nelec)
#    eT = fted_E(1., h1e, eri, norb, nelec, True)
#    print eT
#    gen_rdm1s(h1e, eri, norb, nelec, True)

    rdma, rdmb = fted_rdm1s(2., h1e, eri, norb, nelec)
    print rdma
