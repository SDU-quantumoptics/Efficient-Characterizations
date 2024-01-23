# -*- coding: utf-8 -*-


import numpy as np


def P_lambda(lam1):
    P0 = np.array([[1],[0],[0],[0]], dtype='float64')
    P1 = np.array([[0],[0],[0],[1]], dtype='float64')
    P = [P0, P1]
    p_lamda = P[lam1]
    return p_lamda

psi_l = [np.array([[1],[0],[0],[1]], dtype='float64'),
         np.array([[1],[0],[0],[-1]], dtype='float64'),
         np.array([[1],[1],[0],[0]], dtype='float64'),
         np.array([[1],[-1],[0],[0]], dtype='float64'),
         np.array([[1],[0],[1],[0]], dtype='float64'),
         np.array([[1],[0],[-1],[0]], dtype='float64')] # H,V,+,-,R,L

def f_lambda(lam1, freqs):
    f_ll = []
    for l1 in range(6):
        psi_ll = psi_l[l1]
        f_ll.append(freqs[l1]*(P_lambda(lam1).T @ psi_ll)[0])
    return np.sum(f_ll)

# calculate the inverse of channel M
def inv_channel_M(freqs):
    Pai0 = np.diag([1,0,0,0.])
    Pai1 = np.diag([0,1,1,1.])
    Pai = [Pai0, Pai1]
    M = np.zeros((4,4), dtype='float64')
    for lam1 in range(2):
        f_lam = f_lambda(lam1, freqs)
        Pai_lam = Pai[lam1]
        M += f_lam*Pai_lam

    return np.diag(1/np.diagonal(M))




if __name__ == '__main__':

    freqs = np.array([[1/3], [0], [1/6], [1/6], [1/6], [1/6]])# ideal_frequency for zero state on six bases
    
    M_inver_ideal = inv_channel_M(freqs).astype('complex128')











