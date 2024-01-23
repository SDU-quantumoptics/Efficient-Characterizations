# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 01:12:10 2024

@author: skyx
"""

import numpy as np
import qutip as qt

rho = np.zeros((20, 2,2), dtype='complex128')
for i in range(20):
    rnd_rho = np.array(qt.rand_dm(2, pure=False))
    rho[i,:,:] = rnd_rho
np.save('standard_1q_mix_state.npy', rho)