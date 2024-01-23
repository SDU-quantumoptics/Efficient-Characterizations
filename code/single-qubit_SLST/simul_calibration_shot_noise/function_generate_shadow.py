# -*- coding: utf-8 -*-


# import pennylane as qml
import numpy as np
import random
import qutip as qt
import time
from numba import jit
# from func_basic import rho_2q_t, rho_fidelity



zero_state = np.array([[1.+0.j, 0.+0.j],[0.+0.j, 0.+0.j]], dtype='complex128')
one_state = np.array([[0.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]], dtype='complex128')
# # local qubit unitaries
phase_z = np.array([[1.+0.j, 0.+0.j], [0.+0.j, 0.-1.j]], dtype='complex128')
hadamard = (1/(2**0.5))*np.array([[1.+0.j, 1.+0.j], [1.+0.j, -1.+0.j]], dtype='complex128')
identity = np.array([[1.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]], dtype='complex128')
# # undo the rotations that were added implicitly to the circuit for the Pauli measurements
unitaries = np.array([hadamard, hadamard @ phase_z, identity], dtype='complex128')
    
    

 
#################################################################################generate shadow

sigma_x = np.array([[0., 1.], [1., 0.]],  dtype = 'complex128')
sigma_y = np.array([[0.+0.j, 0.-1.j], [0.+1.j, 0.+0.j]],  dtype = 'complex128')
sigma_z = np.array([[ 1.,  0.], [ 0., -1.]],  dtype = 'complex128')
identity = np.array([[ 1.,  0.], [ 0., 1.]],  dtype = 'complex128')

@jit(nopython=True)
def generate_real_p(rho, sigma1, sigma2):
    sigmai_a = sigma1
    sigmai_b = sigma2
    AB = np.trace(np.dot(np.kron(sigmai_a, sigmai_b), rho))
    AI = np.trace(np.dot(np.kron(sigmai_a, identity), rho))
    IB = np.trace(np.dot(np.kron(identity, sigmai_b), rho))
    II = np.trace(np.dot(np.kron(identity, identity), rho))
    p11 = (AB+AI+IB+II)/4
    p10 = (AI+II-AB-IB)/4
    p01 = (IB+II-AB-AI)/4
    p00 = (AB+II-AI-IB)/4
    return(np.real(np.array([p11, p10, p01, p00])))
    # return(np.array([0.1,0.2,0.3,0.4]))

@jit(nopython=True)
def generate_shot_result(rho, sigma1, sigma2):
    real_p = generate_real_p(rho, sigma1, sigma2)
    p11 = real_p[0]
    p10 = real_p[1]
    p01 = real_p[2]
    
    rnd = random.random()
    if rnd < p11:
        result = 0
    elif rnd < p11+p10:
        result = 1
    elif rnd < p11+p10+p01:
        result = 2
    else:
        result = 3
    return result

unitary_ensemble = np.array([sigma_x, sigma_y, sigma_z], dtype='complex128')

@jit(nopython=True)
def shadow_result(rho, shadow_size, num_qubits=2):

    # sample random Pauli measurements uniformly, where 0,1,2 = X,Y,Z
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))
    outcomes = np.zeros((shadow_size, num_qubits))
    results = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

    for ns in range(shadow_size):
        sigma1 = unitary_ensemble[int(unitary_ids[ns, 0])]
        sigma2 = unitary_ensemble[int(unitary_ids[ns, 1])]
        outcomes[ns, :] = results[generate_shot_result(rho, sigma1, sigma2)]
    # print(sigmas[0, :, :])

    return (outcomes, unitary_ids)

############################################################################generate crosstalk noise shadow
@jit(nopython=True)
def generate_shot_result_crosstalk(rho, sigma1, sigma2, crosstalk):
    real_p = generate_real_p(rho, sigma1, sigma2)
    pctalk1 = real_p[0]*crosstalk
    pctalk0 = real_p[2]*crosstalk
    
    p11 = real_p[0]-pctalk1+pctalk0
    p01 = real_p[2]-pctalk0+pctalk1

    pctalk1 = real_p[1]*crosstalk
    pctalk0 = real_p[3]*crosstalk
    
    p10 = real_p[1]-pctalk1+pctalk0
    p00 = real_p[3]-pctalk0+pctalk1
    # print(p11+p10+p01+p00)
    rnd = random.random()
    if rnd < p11:
        result = 0
    elif rnd < p11+p10:
        result = 1
    elif rnd < p11+p10+p01:
        result = 2
    else:
        result = 3
    return result

unitary_ensemble = np.array([sigma_x, sigma_y, sigma_z], dtype='complex128')

@jit(nopython=True)
def shadow_result_crosstalk(rho, shadow_size, crosstalks, num_qubits=2):

    # sample random Pauli measurements uniformly, where 0,1,2 = X,Y,Z
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))
    outcomes = np.zeros((shadow_size, num_qubits))
    results = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

    for ns in range(shadow_size):
        sigma1 = unitary_ensemble[int(unitary_ids[ns, 0])]
        sigma2 = unitary_ensemble[int(unitary_ids[ns, 1])]
        id1 = int(unitary_ids[ns, 0])
        outcomes[ns, :] = results[generate_shot_result_crosstalk(rho, sigma1, sigma2, crosstalks[id1])]
    # print(sigmas[0, :, :])

    return (outcomes, unitary_ids)


############################################################################generate random U noise shadow
@jit(nopython=True)
def rand_rotationU(phi, theta, delta):
    rn = np.array(
    [[np.cos(delta/2) - (0+1j)*np.cos(theta)*np.sin(delta/2),
      -(0+1j)*np.exp(-(0+1j)*phi)*np.sin(delta/2)*np.sin(theta)],
     [-(0+1j)*np.exp((0+1j)*phi)*np.sin(delta/2)*np.sin(theta),
      np.cos(delta/2) + (0+1j)*np.cos(theta)*np.sin(delta/2)]]
    )
    return rn

@jit(nopython=True)
def generate_real_p_randU(rho, sigma1, sigma2, phis, thetas, delta):
    E_a1 = (sigma1+identity)/2
    E_a0 = (identity-sigma1)/2
    E_b1 = (sigma2+identity)/2
    E_b0 = (identity-sigma2)/2
    
    # pertubate E
    U1 = rand_rotationU(phis[0], thetas[0], delta)
    U0 = rand_rotationU(phis[1], thetas[1], delta)
    E_a1 = U1 @ E_a1 @ U1.T.conj()
    E_a0 = U0 @ E_a0 @ U0.T.conj()
    
    # AB = np.trace(np.dot(np.kron(sigmai_a, sigmai_b), rho))
    # AI = np.trace(np.dot(np.kron(sigmai_a, identity), rho))
    # IB = np.trace(np.dot(np.kron(identity, sigmai_b), rho))
    # II = np.trace(np.dot(np.kron(identity, identity), rho))
    
    p11 = np.trace(np.dot(np.kron(E_a1, E_b1), rho))
    p10 = np.trace(np.dot(np.kron(E_a1, E_b0), rho))
    p01 = np.trace(np.dot(np.kron(E_a0, E_b1), rho))
    p00 = np.trace(np.dot(np.kron(E_a0, E_b0), rho))
    return np.real(np.array([p11, p10, p01, p00])/(p11+p10+p01+p00))

@jit(nopython=True)
def generate_shot_result_randU(rho, sigma1, sigma2, phis, thetas, delta):
    real_p = generate_real_p_randU(rho, sigma1, sigma2, phis, thetas, delta)
    p11 = real_p[0]
    p10 = real_p[1]
    p01 = real_p[2]
    p00 = real_p[3]
    # print(p11+p10+p01+p00)
    rnd = random.random()
    if rnd < p11:
        result = 0
    elif rnd < p11+p10:
        result = 1
    elif rnd < p11+p10+p01:
        result = 2
    else:
        result = 3
    return result

unitary_ensemble = np.array([sigma_x, sigma_y, sigma_z], dtype='complex128')

@jit(nopython=True)
def shadow_result_randU(rho, shadow_size, phi_ls, theta_ls, delta, num_qubits=2):

    # sample random Pauli measurements uniformly, where 0,1,2 = X,Y,Z
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))
    outcomes = np.zeros((shadow_size, num_qubits))
    results = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

    for ns in range(shadow_size):
        sigma1 = unitary_ensemble[int(unitary_ids[ns, 0])]
        sigma2 = unitary_ensemble[int(unitary_ids[ns, 1])]
        id1 = int(unitary_ids[ns, 0])
        outcomes[ns, :] = results[generate_shot_result_randU(rho, sigma1, sigma2, phi_ls[id1,:], theta_ls[id1,:], delta)]
    # print(sigmas[0, :, :])

    return (outcomes, unitary_ids)


###############################################################################################

basis_to_shadow = np.array([[1, -1, 1, -1, 1, -1], [2, 2, 0, 0, 1, 1]]) #[results, sigmas]
shadow_to_basis = np.array([[2,3],[4,5],[0,1]], dtype='int64')
def statistic_shadow(shadow):
    coin_counts = np.zeros((6,6), dtype='int64')
    
    results = shadow[0].astype('int')
    results[results>0] = 0
    sigmas = shadow[1]
    for i in range(len(shadow[1])):
        
        x = shadow_to_basis[sigmas[i, 0], results[i, 0]]
        y = shadow_to_basis[sigmas[i, 1], results[i, 1]]
        coin_counts[x, y] += 1
    return coin_counts/np.sum(coin_counts)



if __name__ == '__main__':
    start = time.time()
    
    total_photon = 1000
    
    # rho_standard = np.array(qt.rand_dm(4, pure=False))
    rho_standard = np.diag([1,0,0,0]).astype('complex128')
    crosstalks = np.array([0.1, 0.1, 0.1])
    shadow = shadow_result_crosstalk(rho_standard, total_photon, crosstalks)
    

    # phi_ls = np.random.uniform(0, 2*np.pi, (3, 2))
    # theta_ls = np.random.uniform(0, np.pi, (3, 2))
    # np.save('randomU_list_phi.npy', phi_ls)
    # np.save('randomU_list_theta.npy', theta_ls)
    phi_ls = np.load('randomU_list_phi.npy')
    theta_ls = np.load('randomU_list_theta.npy')
    delta = 0.3*np.pi
    shadow = shadow_result_randU(rho_standard, total_photon, phi_ls, theta_ls, delta)
    a = statistic_shadow(shadow)

    
    
    
    