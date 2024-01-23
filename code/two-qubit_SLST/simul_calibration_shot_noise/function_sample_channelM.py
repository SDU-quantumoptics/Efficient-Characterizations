# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit


# custom functions
from function_basic import rho_to_paras_mix, rho_2q_t, rho_fidelity, statistic_shadow
from function_robust_shadow import R_shadow_state_reconstruction, ket_to_rho
from function_generate_shadow import shadow_result, statistic_shadow
from function_channelM import inv_channel_M



def sample_particles(your_probability_matrix, num_particles):
    flattened_matrix = your_probability_matrix.flatten()
    sampled_indices = np.random.choice(len(flattened_matrix), size=num_particles, p=flattened_matrix)
    
    
    sampled_distribution = np.zeros_like(flattened_matrix)
    for i in range(num_particles):
        sampled_distribution[sampled_indices[i]] += 1
    
    sampling_result = sampled_distribution.reshape((6, 6))
    return sampling_result/np.sum(sampling_result)



def calibration_and_gain_Minv(cali_probs):
    cali_shots = 2000
    new_cali_freqs = sample_particles(cali_probs, cali_shots)
    M_inv = inv_channel_M(new_cali_freqs)
    return M_inv.astype('complex128')

j=0
cali_probs = np.load('cali_probs_j='+str(j)+'.npy')
new_M = calibration_and_gain_Minv(cali_probs)





