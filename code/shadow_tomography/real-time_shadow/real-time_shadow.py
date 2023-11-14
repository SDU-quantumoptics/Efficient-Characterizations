# -*- coding: utf-8 -*-

import numpy as np
from numba import jit
import qutip as qt


####################################################################################
# real-time shadow tomography

num_qubits = 1 

zero_state = np.array([[1.+0.j, 0.+0.j],[0.+0.j, 0.+0.j]], dtype='complex128') 
one_state = np.array([[0.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]], dtype='complex128')

# local qubit unitaries
phase_z = np.array([[1.+0.j, 0.+0.j], [0.+0.j, 0.-1.j]], dtype='complex128')
hadamard = (1/(2**0.5))*np.array([[1.+0.j, 1.+0.j], [1.+0.j, -1.+0.j]], dtype='complex128')
identity = np.array([[1.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]], dtype='complex128')

unitaries = np.array([hadamard, hadamard @ phase_z, identity], dtype='complex128')

@jit(nopython=True)
def singleshot_state(b_list, obs_list):

    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[int(obs_list[i]), :, :]

        local_rho = 3 * (U.conj().T @ state @ U) - identity
        if i==0:
            rho_shot = local_rho
        else:
            rho_shot = np.kron(rho_shot, local_rho)
        
    return rho_shot



if __name__ == '__main__':
    
    fidelity_real_time = np.zeros((5, 315))
    state = 6 # choose the shadow of which state from 20 states
    num = 0 # we repeat measurement for each state 10 times, choose which time is used
    shots = 315
    
    
    shadow_pauli = np.load('shadow_pauli.npy')
    shadow_outcome = np.load('shadow_outcome.npy')
    single_states = []
    for i in range(shots):
        single_state = singleshot_state(shadow_outcome[state, num, i], shadow_pauli[state, num, i])
        single_states.append(single_state)

    for ob_num in range(5): # randomly choose five observables
        X_ket = np.array(qt.rand_ket_haar(2))
        target_state_dm = X_ket @ X_ket.conj().T 

        fidelity_ls = []
        for j in range(shots):
            shadow_state = np.mean(single_states[:j+1], axis=0)
            fidelity = np.real(np.trace(shadow_state @ target_state_dm))
            fidelity_ls.append(fidelity)
        fidelity_real_time[ob_num, :] = fidelity_ls

    print(fidelity_real_time)
    # np.save('real_time_fidelity_5ob.npy', fidelity_real_time)
    
    
    
