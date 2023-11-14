# -*- coding: utf-8 -*-

# import pennylane as qml
import numpy as np
# import qutip as qt
from numba import jit


sigma_x = np.array([[0., 1.], [1., 0.]],  dtype = complex)
sigma_y = np.array([[0.+0.j, 0.-1.j], [0.+1.j, 0.+0.j]],  dtype = complex)
sigma_z = np.array([[ 1.,  0.], [ 0., -1.]],  dtype = complex)
identity = np.array([[ 1.,  0.], [ 0., 1.]],  dtype = complex)
sigmas = np.array([identity, sigma_x, sigma_y, sigma_z])

@jit(nopython=True)
def to_rho_ket(rho):
    rho_ket = np.zeros((16, 1), dtype = 'complex128')
    for i in range(4):
        for j in range(4):
            rho_ket[i*4+j, 0] = np.trace(np.kron(sigmas[i], sigmas[j]) @ rho)/2
    return rho_ket
    
    
    
# computational basis states
state_0 = np.array([[1.+0.j, 0.+0.j],[0.+0.j, 0.+0.j]])
state_1 = np.array([[0.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]])
state_00 = np.kron(state_0, state_0)
state_01 = np.kron(state_0, state_1)
state_10 = np.kron(state_1, state_0)
state_11 = np.kron(state_1, state_1)
states = [state_00, state_01, state_10, state_11]

b_ket = np.zeros((4, 16, 1), dtype = complex)
for i in range(4):
    b_ket[i, :, :] = to_rho_ket(states[i])
    
#load U
u_sigma = np.load('u_sigma.npy')
    
#load M^-1
M_inver = np.zeros((5, 16, 16), dtype = complex)
for j in range(5):
    M_inver[j, :, :] = np.loadtxt('j='+str(j)+'.txt')


################################################ robust shadow tomography
@jit(nopython=True)
def R_snapshot_state(result, sigma, noise_j):
    if result[0]>0:
        if result[1]>0:
            b_state = b_ket[0]
        else:
            b_state = b_ket[1]
    
    else:
        if result[1]>0:
            b_state = b_ket[2]
        else:
            b_state = b_ket[3]
    
    sigma_index = int(sigma[0]*3+sigma[1])
    random_index = np.random.randint(16)
    U = u_sigma[sigma_index, random_index, :, :]
    rho_snapshot = np.dot(M_inver[noise_j, :, :], np.dot(U.conj().T, b_state))
            
    return rho_snapshot


# reconstruct rho_hat with robust shadow tomography
def R_shadow_state_reconstruction(shadow, noise_j):
    num_qubits = 2

    num_snapshots, num_qubits = shadow[0].shape

    # classical values
    results, sigmas = shadow

    # Averaging over snapshot states.
    shadow_rho = np.zeros((16, 1), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += R_snapshot_state(results[i], sigmas[i], noise_j)
        # print(shadow_rho)
    return shadow_rho / num_snapshots



# iden = np.array(qt.identity(2), dtype='complex128')
# sigmax = np.array(qt.sigmax(), dtype='complex128')
# sigmay = np.array(qt.sigmay(), dtype='complex128')
# sigmaz = np.array(qt.sigmaz(), dtype='complex128')
# sigmas = np.array([iden, sigmax, sigmay, sigmaz])
def ket_to_rho(stokes):
    rho = np.zeros((4, 4), dtype='complex128')
    for i in range(4):
        for j in range(4):
            rho += np.kron(sigmas[i,:,:], sigmas[j,:,:])* stokes[i*4+j]
            
    return rho/2


    
    
    
    