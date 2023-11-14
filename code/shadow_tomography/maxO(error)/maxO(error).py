# -*- coding: utf-8 -*-

import numpy as np
from numba import jit
import qutip as qt
import matplotlib.pyplot as plt


####################################################################################
#shadow tomoraphy

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
    
    sn = 6 # choose which state of 20 states to be the target state
    
    error_all = np.zeros((10, 21, 128)) # 10 repeats, 315 photons total
    shots = np.linspace(15, 315, 21, dtype='int32') # used photon from 15 to 315

    shadow_pauli = np.load('shadow_pauli.npy').reshape((20, 3150,1)) # load shadow
    shadow_outcome = np.load('shadow_outcome.npy').reshape((20, 3150,1)) # load shadow
    
    ST = np.load('standard_20states.npy') # load ideal state
    ket = ST[sn, :, :]
    rho_standard = ket @ ket.conj().T
    
    # generate 128 observables randomly
    X_ls = []
    for X_n in range(128):
        X_ket = np.array(qt.rand_ket_haar(2))
        X = X_ket @ X_ket.conj().T 
        X_ls.append(X)
        
    
    # start calculate maxO(error)
    for repeat in range(10):
        i_index = np.arange(3150)
        np.random.shuffle(i_index)
        
        single_states = []
        for i in i_index[:315]:
            
            single_state = singleshot_state(shadow_outcome[sn, i], shadow_pauli[sn, i])
            single_states.append(single_state)
            
        for j in range(21):
            shadow_state = np.mean(single_states[:shots[j]+1], axis=0)
            for X_n in range(128):
                fidelity = np.real(np.trace(shadow_state @ X_ls[X_n]))
                real_fid = np.real(np.trace(rho_standard @ X_ls[X_n]))
                error_all[repeat, j, X_n] = np.abs(fidelity - real_fid)


    
    plt.figure()
    plt.plot(shots, np.mean(np.max(error_all[:, :, :], axis=2), axis=0))
    plt.errorbar(shots, np.mean(np.max(error_all[:, :, :], axis=2), axis=0), 
                 yerr=np.std(np.max(error_all[:, :, :], axis=2), axis=0))

    # np.save('max_error_1state.npy', error_all)


















