# -*- coding: utf-8 -*-
# import pennylane as qml
import numpy as np
from numba import jit



zero_state = np.array([[1.+0.j, 0.+0.j],[0.+0.j, 0.+0.j]], dtype='complex128')
one_state = np.array([[0.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]], dtype='complex128')
# # local qubit unitaries
phase_z = np.array([[1.+0.j, 0.+0.j], [0.+0.j, 0.-1.j]], dtype='complex128')
hadamard = (1/(2**0.5))*np.array([[1.+0.j, 1.+0.j], [1.+0.j, -1.+0.j]], dtype='complex128')
identity = np.array([[1.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]], dtype='complex128')
# # undo the rotations that were added implicitly to the circuit for the Pauli measurements
unitaries = np.array([hadamard, hadamard @ phase_z, identity], dtype='complex128')
    
    
@jit(nopython=True)
def snapshot_state(b_list, obs_list):

    num_qubits = len(b_list)
    
    
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[int(obs_list[i]), :, :]

        local_rho = 3 * np.dot(U.conj().T, np.dot(state, U)) - identity
        
        if i==0:
            rho_snapshot = local_rho
        else:
            rho_snapshot = np.kron(rho_snapshot, local_rho)
            
    # return num_qubits
    return rho_snapshot



@jit(nopython=True)
def shadow_state_reconstruction(result, sigma):

    """
    Reconstruct a state approximation as an average over all snapshots in the shadow.

    Args:
        shadow (tuple): A shadow tuple obtained from `calculate_classical_shadow`.

    Returns:
        Numpy array with the reconstructed quantum state.
    """
    num_snapshots, num_qubits = result.shape

    # classical values
    b_lists, obs_lists = result, sigma

    # Averaging over snapshot states.
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype='complex128')
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])
        # print(shadow_rho)
    return shadow_rho / num_snapshots


    
    
    
    