# -*- coding: utf-8 -*-

import numpy as np
from numba import jit



@jit(nopython=True)
def snapshot_state(b_list, obs_list):

    num_qubits = len(b_list)
    
    # computational basis states
    zero_state = np.array([[1.+0.j, 0.+0.j],[0.+0.j, 0.+0.j]])
    one_state = np.array([[0.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]])
    
    # # local qubit unitaries
    phase_z = np.array([[1.+0.j, 0.+0.j], [0.+0.j, 0.-1.j]])
    hadamard = (1/(2**0.5))*np.array([[1.+0.j, 1.+0.j], [1.+0.j, -1.+0.j]])
    identity = np.array([[1.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]])
    
    # # undo the rotations that were added implicitly to the circuit for the Pauli measurements
    unitaries = [hadamard, hadamard @ phase_z, identity]

    # # reconstructing the snapshot state from local Pauli measurements
    # rho_snapshot = [1]
    
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[int(obs_list[i])]

        local_rho = 3 * (U.conj().T @ state @ U) - identity
        
        if i==0:
            rho_snapshot = local_rho
        else:
            rho_snapshot = np.kron(rho_snapshot, local_rho)
    # return num_qubits
    return rho_snapshot

def shadow_state_reconstruction(shadow):

    num_snapshots, num_qubits = shadow[0].shape

    # classical values
    b_lists, obs_lists = shadow

    # Averaging over snapshot states.
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])
       

    return shadow_rho / num_snapshots




    
    
    
    
    
    
    