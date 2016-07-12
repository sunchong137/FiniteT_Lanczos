#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Adeline C. Sun <chongs@princeton.edu>
#

import numpy
import pyscf.lib
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
import ftlanczos as flan
from smpl import smpl_hilbert as ftsmpl

def kernel(h1e, g2e, norb, nelec):

    h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec//2
    else:
        neleca = nelec[0]

    na = cistring.num_strings(norb, neleca)
    ci0 = numpy.zeros((na,na))
    ci0[0,0] = 1

    def hop(c):
        hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    hdiag = direct_spin1.make_hdiag(h1e, g2e, norb, nelec)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = pyscf.lib.davidson(hop, ci0.reshape(-1), precond)
    return e, c

def kernel_ft(h1e, g2e, norb, nelec, T, m=50, nsamp=100, Tmin=10e-4):
    '''E at temperature T
    '''
    if T < Tmin:
        e, c = kernel(h1e, g2e, norb, nelec)
        return e
    h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    def vecgen(n1=na, n2=nb):
        ci0 = numpy.random.randn(n1, n2)
        return ci0.reshape(-1)
    def hop(c):
        hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    E = flan.ftlan_E(hop, vecgen, T, m, nsamp)
    return E

def kernel_ft_smpl(h1e, g2e, norb, nelec, T, vecgen = 0, m=50, nsmpl = 250, nblk = 10, Tmin=10e-4, nrotation = 200):
    if T < Tmin:
        e, c = kernel(h1e, g2e, norb, nelec)
        return e
    disp = numpy.exp(T) * 0.1 # displacement
    h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = numpy.random.randn(na, nb)
    def hop(c):
        hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)

    E, dev, ar = ftsmpl(hop, ci0, T, flan.ftlan_E1c, nsamp = nsmpl, dr = disp, genci=vecgen, nblock = nblk, nrot = nrotation)
    # ar is the acceptance ratio
    return E, dev, ar

   
def ft_rdm1s(h1e, g2e, norb, nelec, T, m=50, nsamp=40, Tmin=10e-4):
    '''rdm of spin a and b at temperature T
    '''
    if T < Tmin:
       e, c = kernel(h1e, g2e, norb, nelec)
       rdma, rdmb = direct_spin1.make_rdm1s(c, norb, nelec)
       return rdma, rdmb

    h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    def vecgen(n1=na, n2=nb):
        ci0 = numpy.random.randn(n1, n2)
#        ci0[0, 0] = 1.
        return ci0.reshape(-1)
    def hop(c):
        hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    def qud(v1, v2):
        dma, dmb = direct_spin1.trans_rdm1s(v1, v2, norb, nelec)
        return dma, dmb

#    rdma, rdmb = flan.ht_rdm1s(qud, hop, vecgen, T, norb, m, nsamp)
    rdma, rdmb = flan.ftlan_rdm1s(qud, hop, vecgen, T, norb, m, nsamp)
    return rdma, rdmb

def ft_rdm1(h1e, g2e, norb, nelec, T, m=50, nsamp=40):
    rdma, rdmb = ft_rdm1s(h1e, g2e, norb, nelec, T, m, nsamp)
    return rdma+rdmb

if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo

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
    ne = mol.nelectron - 2
    nelec = (nelec//2, nelec-nelec//2)
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)
    e1, ci0 = kernel(h1e, eri, norb, ne) #FCI kernel
     

    print "T = 0, E = ", e1
    
    rdma0, rdmb0 = direct_spin1.make_rdm1s(ci0, norb, nelec)
    print "*********************"
    print "zero rdm:\n", rdma0, "\n", rdmb0
    print "*********************"
    rdma, rdmb = ft_rdm1s(h1e, eri, norb, nelec, 10., 10, 10)
    print rdma, "\n", rdmb
#    print numpy.sum(numpy.diag(rdma))
#    print "T = 0, E = %10.10f"%e1
    
    e2 = kernel_ft(h1e, eri, norb, nelec, 0.1, 40, 20)
    print "E(T) = %10.10f"%e2
    e3 = kernel_ft_smpl(h1e, eri, norb, nelec, 0.1, )
    print "E(T)_smpl = %10.10f"%e3 
'''
    f = open("data/E-T_3.dat", "w")
    f.write("%2.4f       %2.10f\n"%(0., e1))
    for i in range(30):
        T = 0.1+0.2*i
        e2 = kernel_ft(h1e, eri, norb, nelec, T) # finite FCI kernel
        print "%2.4f       %2.10f"%(T, e2)
        f.write("%2.4f       %2.10f\n"%(T, e2))
    f.close()
'''
