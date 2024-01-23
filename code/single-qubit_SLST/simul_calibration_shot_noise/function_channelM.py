# -*- coding: utf-8 -*-


import numpy as np


def P_lambda(lam1, lam2):
    P0 = np.array([[1],[0],[0],[0]], dtype='float64')
    P1 = np.array([[0],[0],[0],[1]], dtype='float64')
    P = [P0, P1]
    p_lamda = np.kron(P[lam1], P[lam2])
    return p_lamda

psi_l = [np.array([[1],[0],[0],[1]], dtype='float64'),
         np.array([[1],[0],[0],[-1]], dtype='float64'),
         np.array([[1],[1],[0],[0]], dtype='float64'),
         np.array([[1],[-1],[0],[0]], dtype='float64'),
         np.array([[1],[0],[1],[0]], dtype='float64'),
         np.array([[1],[0],[-1],[0]], dtype='float64')] # H,V,+,-,R,L

def f_lambda(lam1, lam2, freqs):
    f_ll = []
    for l1 in range(6):
        for l2 in range(6):
            psi_ll = np.kron(psi_l[l1], psi_l[l2])
            f_ll.append(freqs[l1, l2]*(P_lambda(lam1, lam2).T @ psi_ll)[0,0])
    return np.sum(f_ll)


def inv_channel_M(freqs):
    Pai0 = np.diag([1,0,0,0.])
    Pai1 = np.diag([0,1,1,1.])
    Pai = [Pai0, Pai1]
    M = np.zeros((16,16), dtype='float64')
    for lam1 in range(2):
        for lam2 in range(2):
            f_lam = f_lambda(lam1, lam2, freqs)
            Pai_lam = np.kron(Pai[lam1], Pai[lam2])
            M += f_lam*Pai_lam

    return np.diag(1/np.diagonal(M))




if __name__ == '__main__':

    freqs = np.array([[1/3], [0], [1/6], [1/6], [1/6], [1/6]])
    freqs = np.kron(freqs, freqs.T)
    b = f_lambda(1, 0, freqs)
    
    c = inv_channel_M(freqs)
    # a = np.load('pure_M.npy')











