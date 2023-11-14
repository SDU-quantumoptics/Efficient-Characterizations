# -*- coding: utf-8 -*-

import QuantumTomography as qKLib
import numpy as np

def k_tomo(S):
    H, V, h1, h2, R, L = S[0],S[1],S[2],S[3],S[4],S[5]
    t = qKLib.Tomography()
    t.importConf('config.txt')
    tomo_input = np.array([[1,0,H,1,0],[1,0,V,0,1],[1,0,h1,0.7071,0.7071],[1,0,h2,0.7071,-0.7071],[1,0,R,0.7071,0.7071j],[1,0,L,0.7071,-0.7071j]])
    intensity = np.array([1,1,1,1,1,1])
    rho = t.state_tomography(tomo_input, intensity)[0]
    # print(qKLib.purity(rho))
    return rho