# -*- coding: utf-8 -*-
from numba import jit
import numpy as np
from scipy.linalg import sqrtm

@jit(nopython=True)
def proj_ob(ob, rho):
    s = np.real(np.trace(ob @ rho))
    return np.real(s)

def rho_fidelity(rho1, rho2):
    sqr_rho1 = sqrtm(rho1)
    fid = ((sqrtm(sqr_rho1 @ rho2 @ sqr_rho1)).trace())
    return(np.real(fid))

@jit(nopython=True)
def rho_nq(params, n):
    # n: total parameters
    para_real = int((n+1)/2)
    amp = params[:para_real]
    phi = params[para_real:]
    
    norm = np.sqrt(np.sum(amp**2))
    amps = amp/norm
    phis = np.exp(np.append([0,], phi)*(0+1j))
    
    ket = np.zeros((para_real, 1), dtype = 'complex128')
    ket[:, 0] = amps*phis

    return ket @ ket.conj().T

    
def shadow_to_tomo_inpt(shadow):
    outcome = ((shadow[0]+1)/2).astype('int32')
    sigma = shadow[1].astype('int32')
    statistic = np.zeros(6 ,dtype='int32')
    bases_index = np.array([[3,5,1],[2,4,0]])
    
    for i in range(len(outcome[:,0])):
        index = bases_index[outcome[i,0], sigma[i,0]]
        statistic[index] += 1
    return statistic


    

    

        











