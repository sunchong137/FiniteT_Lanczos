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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import pyscf.lib
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
import ftlanczos as flan
from ftlanczos import ftlan_E
from ftlanczos import ftlan_rdm1s

def contract_1e(f1e, fcivec, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    fcivec = fcivec.reshape(na,nb)
    t1 = numpy.zeros((norb,norb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * fcivec[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1[a,i,:,str1] += sign * fcivec[:,str0]
    fcinew = numpy.dot(f1e.reshape(-1), t1.reshape(-1,na))
    return fcinew


def contract_2e(eri, fcivec, norb, nelec, opt=None):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    fcivec = fcivec.reshape(na,nb)
    t1 = numpy.zeros((norb,norb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * fcivec[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1[a,i,:,str1] += sign * fcivec[:,str0]
    t1 = numpy.dot(eri.reshape(norb*norb,-1), t1.reshape(norb*norb,-1))
    t1 = t1.reshape(norb,norb,na,nb)
    fcinew = numpy.zeros_like(fcivec)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t1[a,i,str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:,str1] += sign * t1[a,i,:,str0]
    return fcinew


def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    '''Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
    '''
    if not isinstance(nelec, (int, numpy.integer)):
        nelec = sum(nelec)
    eri = eri.copy()
    h2e = pyscf.ao2mo.restore(1, eri, norb)
    f1e = h1e - numpy.einsum('jiik->jk', h2e) * .5
    f1e = f1e * (1./(nelec+1e-100))
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    return h2e * fac


def make_hdiag(h1e, g2e, norb, nelec, opt=None):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    occslista = [tab[:neleca,0] for tab in link_indexa]
    occslistb = [tab[:nelecb,0] for tab in link_indexb]
    g2e = ao2mo.restore(1, g2e, norb)
    diagj = numpy.einsum('iijj->ij',g2e)
    diagk = numpy.einsum('ijji->ij',g2e)
    hdiag = []
    for aocc in occslista:
        for bocc in occslistb:
            e1 = h1e[aocc,aocc].sum() + h1e[bocc,bocc].sum()
            e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
               + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
               - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
            hdiag.append(e1 + e2*.5)
    return numpy.array(hdiag)

def kernel(h1e, g2e, norb, nelec):

    h2e = absorb_h1e(h1e, g2e, norb, nelec, .5)

    na = cistring.num_strings(norb, nelec//2)
    ci0 = numpy.zeros((na,na))
    ci0[0,0] = 1

    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    hdiag = make_hdiag(h1e, g2e, norb, nelec)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = pyscf.lib.davidson(hop, ci0.reshape(-1), precond)
    return e, c

#def kernel_ft_old(h1e, g2e, norb, nelec, T): # finite temperature
#   h2e = absorb_h1e(h1e, g2e, norb, nelec, .5)
#   na = cistring.num_strings(norb, nelec//2)
#   ci0 = 10e-15*numpy.random.rand(na, na) # small deviation
#   ci0[0,0] = 1.
#   ci0 = ci0/numpy.linalg.norm(ci0)

#   def hop(c):
#       hc = contract_2e(h2e, c, norb, nelec)
#       return hc.reshape(-1)

#   E = ftl_e(hop, ci0.reshape(-1), T)
#   return E

def kernel_ft(h1e, g2e, norb, nelec, T, m=50, nsamp=40):
    '''E at temperature T
    '''
    h2e = absorb_h1e(h1e, g2e, norb, nelec, .5)
    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    def vecgen(n1=na, n2=nb):
        ci0 = numpy.random.rand(n1, n2)
        return ci0.reshape(-1)
    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)

    E = ftlan_E(hop, vecgen, T, m, nsamp)
    return E

def ft_rdm1s(h1e, g2e, norb, nelec, T, m=50, nsamp=40):
    '''rdm of spin a and b at temperature T
    '''
    h2e = absorb_h1e(h1e, g2e, norb, nelec, .5)
    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    def vecgen(n1=na, n2=nb):
        ci0 = numpy.random.rand(n1, n2)
#        ci0[0, 0] = 1.
        return ci0.reshape(-1)
    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec)
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

# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec//2)
    na = cistring.num_strings(norb, nelec//2)
    fcivec = fcivec.reshape(na,na)
    rdm1 = numpy.zeros((norb,norb))
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in link_index[str0]:
            rdm1[a,i] += sign * numpy.dot(fcivec[str1],fcivec[str0])
    for str0, tab in enumerate(link_index):
        for k in range(na):
            for a, i, str1, sign in link_index[str0]:
                rdm1[a,i] += sign * fcivec[k,str1]*fcivec[k,str0]
    return rdm1

# dm_pq,rs = <|p^+ q r^+ s|>
def make_rdm12(fcivec, norb, nelec, opt=None):
    link_index = gen_linkstr_index(range(norb), nelec//2)
    na = num_strings(norb, nelec//2)
    fcivec = fcivec.reshape(na,na)

    rdm1 = numpy.zeros((norb,norb))
    rdm2 = numpy.zeros((norb,norb,norb,norb))
    for str0, tab in enumerate(link_index):
        t1 = numpy.zeros((na,norb,norb))
        for a, i, str1, sign in link_index[str0]:
            for k in range(na):
                t1[k,i,a] += sign * fcivec[str1,k]

        for k, tab in enumerate(link_index):
            for a, i, str1, sign in tab:
                t1[k,i,a] += sign * fcivec[str0,str1]

        rdm1 += numpy.einsum('m,mij->ij', fcivec[str0], t1)
        # i^+ j|0> => <0|j^+ i, so swap i and j
        rdm2 += numpy.einsum('mij,mkl->jikl', t1, t1)
    return reorder_rdm(rdm1, rdm2)


def reorder_rdm(rdm1, rdm2):
    '''reorder from rdm2(pq,rs) = <E^p_q E^r_s> to rdm2(pq,rs) = <e^{pr}_{qs}>.
    Although the "reoredered rdm2" is still in Mulliken order (rdm2[e1,e1,e2,e2]),
    it is the right 2e DM (dotting it with int2e gives the energy of 2e parts)
    '''
    nmo = rdm1.shape[0]
    if inplace:
        rdm2 = rdm2.reshape(nmo,nmo,nmo,nmo)
    else:
        rdm2 = rdm2.copy().reshape(nmo,nmo,nmo,nmo)
    for k in range(nmo):
        rdm2[:,k,k,:] -= rdm1
    return rdm1, rdm2


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
    nelec = mol.nelectron# - 2
    ne = mol.nelectron# - 2
    nelec = (nelec//2, nelec-nelec//2)
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)
    e1, ci0 = kernel(h1e, eri, norb, ne) #FCI kernel
     
#   print "T = 0, E = ", e1
    rdma0, rdmb0 = direct_spin1.make_rdm1s(ci0, norb, nelec)
    print "*********************"
    print "zero rdm:\n", rdma0, "\n", rdmb0
    print "*********************"
    rdma, rdmb = ft_rdm1s(h1e, eri, norb, nelec, 10., 10, 10)
    print rdma, "\n", rdmb
    print numpy.sum(numpy.diag(rdma))
#    print "T = 0, E = %10.10f"%e1

#    e2 = kernel_ft(h1e, eri, norb, nelec, 0.01, 50, 20)
#    print "E(T) = %10.10f"%e2
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
