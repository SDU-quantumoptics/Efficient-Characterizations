# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as la
from scipy.linalg import sqrtm
from numba import jit # speed up the code

# the model of 2qubit mixed states, given ts, we can determine a mixed state
@jit(nopython=True)
def rho_2q_t(ts, qubit_num=2):
    
    d = 2**qubit_num
    Tt = np.zeros((d, d), dtype='complex128')
    for i in range(d):
        Tt[i,i] = ts[i]
    
    b = 0
    for j in range(d-1):
        for k in range(j+1):
            Tt[j+1, k] = ts[b+d] + 1j*ts[b+d+1]
            b += 2

    # rho1 = np.dot(Tt.T.conjugate(), Tt) / np.trace(np.dot(Tt.T.conjugate(), Tt))
    rho2 = np.dot(Tt, Tt.T.conjugate()) / np.trace(np.dot(Tt, Tt.T.conjugate()))
    return rho2


# the model of n-qubit pure states, given params, we can determine a pure state
def rho_nq(params, n):
    # n: total parameters
    para_real = int((n+1)/2)
    re_part = params[:para_real]
    im_part = params[para_real:]
    
    norm = np.sqrt(np.sum(params**2))
    re_part = re_part/norm
    im_part = im_part/norm
    im_part = np.append(im_part, [0+0j,])
    
    ket = np.zeros((para_real, 1), dtype = 'complex128')
    ket[:, 0] = re_part+im_part*1j
    # print(ket)
    return ket @ ket.conj().T

# calculate fidelity of two states
def rho_fidelity(rho1, rho2):
    sqr_rho1 = sqrtm(rho1)
    
    # print(rho1, rho2)
    fid = ((sqrtm(sqr_rho1 @ rho2 @ sqr_rho1)).trace())
    return(np.real(fid))

# make the rhog with negetive eigenvalue to be positive
def make_positive(rhog_in):
    d, v = np.linalg.eig(rhog_in)
    # if d.any()<0:
    #     print(d)
    rhog = np.zeros(rhog_in.shape)
    for j in range(len(d)):
        rhog = rhog + np.abs(d[j])*np.outer(v[:, j], v[:, j].conj().transpose())

    rhog = (rhog + rhog.conj().transpose())/2.0

    return rhog / np.trace(rhog)


def density_to_ket(rho):
    n, v = la.eig(rho)
    index = np.argsort(np.abs(n))
    # for i in range(len(n)):
    #     if abs(n[i]-1.0)<=0.001:
    ket = v[:, index[-1]]
    
    
    norm_ket = ket/ket[-1]
    norm_ket = norm_ket/np.sqrt(np.sum(np.abs(norm_ket)**2))
    # print('v', norm_ket)
    return norm_ket


def rho_to_paras_pure(rho):
    ket = density_to_ket(rho)
    re_part = np.real(ket)
    im_part = np.imag(ket)
    return np.append(re_part, im_part)


def rho_to_paras_mix(rho, qubit_num):
    rho = make_positive(rho)
    Tt_de = la.cholesky(rho, lower=True)
    
    d = 2**qubit_num
    
    ts = np.zeros(4**qubit_num)
    for i in range(d):
        ts[i] = np.real(Tt_de[i,i])
    
    b = 0
    for j in range(d-1):
        for k in range(j+1):
            # Tt[j+1, k] = ts[b+d] + 1j*ts[b+d+1]
            ts[b+d] = np.real(Tt_de[j+1, k])
            ts[b+d+1] = np.imag(Tt_de[j+1, k])
            b += 2
    
    return ts


basis_to_shadow = np.array([[1, -1, 1, -1, 1, -1], [2, 2, 0, 0, 1, 1]]) #[results, sigmas]
shadow_to_basis = np.array([[2,3],[4,5],[0,1]], dtype='int64')
def statistic_shadow(shadow):
    coin_counts = np.zeros((6,6), dtype='int64')
    
    results = shadow[0].astype('int')
    results[results>0] = 0
    sigmas = shadow[1]
    for i in range(len(shadow[1])):
        
        x = shadow_to_basis[sigmas[i, 0], results[i, 0]]
        y = shadow_to_basis[sigmas[i, 1], results[i, 1]]
        coin_counts[x, y] += 1
    return coin_counts



 
