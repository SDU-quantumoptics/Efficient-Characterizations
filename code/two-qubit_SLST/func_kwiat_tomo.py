# -*- coding: utf-8 -*-

import QuantumTomography as qKLib
import numpy as np

# MLE method from Kwiat Quantum Information Group
def k_tomo_2q(counts):
    # counts: array like (6, 6)
    t = qKLib.Tomography()
    t.importConf('2qubit_config.txt')
    tomo_input = get_tomo_input(counts)
    intensity = np.array([1,]*36)
    results = t.state_tomography(tomo_input, intensity)
    # print('purity=', qKLib.purity(results[0]))
    return results[0]

def get_tomo_input(counts):
    # counts: array like 6*6
    bases = [[1,0], [0,1], [0.7071,0.7071], [0.7071,-0.7071], [0.7071,0.7071j], [0.7071,-0.7071j]]
    tomo_bases = []
    for i in range(6):
        for j in range(6):
            tomo_bases.append([1,0,0, counts[i, j]]+bases[i]+bases[j])
    return np.array(tomo_bases)


if __name__ == '__main__':
    counts = np.zeros((6, 6))+5000
    tomo_input = get_tomo_input(counts)