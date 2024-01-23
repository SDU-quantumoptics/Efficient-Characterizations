# -*- coding: utf-8 -*-


# import pennylane as qml
import numpy as np
import random
import qutip as qt
import time
from numba import jit
# from func_basic import rho_2q_t, rho_fidelity




#################################################################################measure fuction

sigma_x = np.array([[0., 1.], [1., 0.]],  dtype = complex)
sigma_y = np.array([[0.+0.j, 0.-1.j], [0.+1.j, 0.+0.j]],  dtype = complex)
sigma_z = np.array([[ 1.,  0.], [ 0., -1.]],  dtype = complex)
identity = np.array([[ 1.,  0.], [ 0., 1.]],  dtype = complex)

@jit(nopython=True)
def generate_p(rho, sigma):
    p = np.real((np.trace(np.dot(sigma, rho))+1))/2
    return p

# @jit(nopython=True)
def generate_shot_result_noise(rho, sigmas, d_mean, sigma_std):
    rho_noise = noise_channel(rho, d_mean, sigma_std)
    
    real_p = generate_p(rho_noise, sigmas)
    
    rnd = random.random()
    if rnd < real_p:
        result = 0
    else:
        result = 1
    return result

def shadow_result(rho, shadow_size, d_mean, sigma_std):
    
    
    """
    Given a state rho, creates a collection of snapshots consisting of a bit string
    and the index of a unitary operation.
    Args:
        shadow_size (int): The number of snapshots in the shadow.
        num_qubits (int): The number of qubits in the circuit.
    """
    # applying the single-qubit Clifford circuit is equivalent to measuring a Pauli
    unitary_ensemble = [sigma_x, sigma_y, sigma_z]

    # sample random Pauli measurements uniformly, where 0,1,2 = X,Y,Z
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, 1))
    outcomes = np.zeros((shadow_size, 1))
    results = [1, -1]

    for ns in range(shadow_size):
        sigma = unitary_ensemble[int(unitary_ids[ns, 0])]
        outcomes[ns, :] = results[generate_shot_result_noise(rho, sigma, d_mean, sigma_std)]
    # print(outcomes)

    return (outcomes, unitary_ids)
    

    
    
def statistic_shadow(shadow):
    outcome = ((shadow[0]+1)/2).astype('int32')
    sigma = shadow[1].astype('int32')
    statistic = np.zeros(6 ,dtype='int32')
    bases_index = np.array([[3,5,1],[2,4,0]])
    
    for i in range(len(outcome[:,0])):
        index = bases_index[outcome[i,0], sigma[i,0]]
        statistic[index] += 1
    return statistic/np.sum(statistic)

normalize = 3
@jit(nopython=True)
def random_deltas(mean, sigma):
    for i in range(1000):
        deltas = np.random.normal(loc=mean*normalize, scale=sigma*normalize, size=3)
        if np.min(deltas)>0:
            # print(i)
            break
    return deltas

sigma_x = np.array([[0., 1.], [1., 0.]],  dtype = complex)
sigma_y = np.array([[0.+0.j, 0.-1.j], [0.+1.j, 0.+0.j]],  dtype = complex)
sigma_z = np.array([[ 1.,  0.], [ 0., -1.]],  dtype = complex)
identity = np.array([[ 1.,  0.], [ 0., 1.]],  dtype = complex)
sigmas = np.array([identity, sigma_x, sigma_y, sigma_z])
def noise_channel(rho, d_mean, sigma_std):
    deltas = random_deltas(d_mean, sigma_std)
    noise_rho = (1-np.sum(deltas)/3)*rho + deltas[0]*sigma_x @ rho @ sigma_x/3 + deltas[1]*sigma_y @ rho @ sigma_y/3 + deltas[2]*sigma_z @ rho @ sigma_z/3
    return noise_rho




    
    
    