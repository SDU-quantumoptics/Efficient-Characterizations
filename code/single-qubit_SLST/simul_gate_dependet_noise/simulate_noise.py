# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit


# custom functions
from function_basic import rho_to_paras_mix, rho_2q_t, rho_fidelity, statistic_shadow
from function_robust_shadow import R_shadow_state_reconstruction, ket_to_rho
from function_generate_shadow import shadow_result, statistic_shadow, noise_channel
from function_channelM import inv_channel_M


time_start = time.time()


############################################ start iteration


@jit(nopython=True)
def coff_delta(BD1, BD2, delta):
    qubit_num = 1
    para_num = 4**qubit_num

    deltas = np.zeros(para_num, dtype='float64')
    
    dw = np.ones(para_num)
    option = np.array([-delta, delta], dtype='float64')
    deltas = np.random.choice(option, para_num)*dw

    BD1 += deltas
    BD2 -= deltas
        
    return BD1, BD2, deltas

# calculate the second and third term of squared F-norm
@jit(nopython=True)
def Fnorm_23(rho_shadow, sigma):
    Fnorm = np.real(np.trace(sigma @ sigma) - 2*np.trace(rho_shadow @ sigma))
    return Fnorm

# SLST function
def SLST_infid(rho_standard, rho_shadow):
    s = 0.602
    t = 0.101
    A = 0
    Δ = 1

    # a = 12.25
    # b = 0.35

    a = 8.533*4 # better guess hyperparameters
    b = 1.421*4
    
    qubit_num = 1
    para_num = 4**qubit_num
    infids = np.ones((repeat, iteration_number))
    rho_para = np.zeros((repeat, para_num)) # with para, we can reconstruct state using mixed state model
    
    for i in range(repeat):
        # 2.set a measurement basis B
        # B_coff = [1,]*para_num
        # B_coff = np.array(B_coff, dtype='float64') # use this: bad guess
        B_coff = rho_to_paras_mix(rho_shadow.copy(), qubit_num=1)+0.2 # use this: good guess
        
        for itr in range(iteration_number):

            k = itr
            βk = b/((k+1)**t)
                
            # 3.calculate B+delta, B-delta
            BD1 = np.array(B_coff)
            BD2 = np.array(B_coff)
            BD_paras = coff_delta(BD1, BD2, βk*Δ)
            BD1_para = BD_paras[0]
            BD2_para = BD_paras[1]
            
            # 4.calculate f(+), f(-)
            ob1 = rho_2q_t(BD1_para, qubit_num)
            ob2 = rho_2q_t(BD2_para, qubit_num)
            
            # f1 = rho_fidelity(ob1, rho_shadow)
            # f2 = rho_fidelity(ob2, rho_shadow)
            f1 = Fnorm_23(rho_shadow, ob1)
            f2 = Fnorm_23(rho_shadow, ob2)
            
            p1 = f1
            p2 = f2

            # 5.calculate gradient gk
            αk = a/((k+1+A)**s)
            gk = (p2-p1)*Δ/(2*βk)
    
            # 6.set B2 = B1+αk*gk
            B_coff = B_coff + BD_paras[2]*gk*αk
            
            # 7.print data
            rho = rho_2q_t(B_coff, qubit_num)
            
            fid = rho_fidelity(rho, rho_standard)
            
            infids[i, itr] = 1.0-fid
            rho_para[i, :] = B_coff
        # print('%.5f' %fid, '\t {:.2%}'.format((i+1)/repeat))
    return infids, rho_para


    
def calibration_and_gain_Minv(cali_shots, delta_mean, sigma_std):
    zero_state = np.diag([1,0.]).astype('complex128') #prepare 0 state
    cali_shadow = shadow_result(zero_state, cali_shots, delta_mean, sigma_std)
    freqs = statistic_shadow(cali_shadow)
    M_inv = inv_channel_M(freqs)
    return M_inv.astype('complex128')
    
    
def state_tomography(method, M_inver):
    
    shadow = shadow_origin
    
    # robust SLST method
    if method == 'RSLST':
        shadow_ket = R_shadow_state_reconstruction(shadow, M_inver)
        shadow_rho = ket_to_rho(shadow_ket)
        infid_itr, BDs = SLST_infid(rho_standard, shadow_rho)
        return infid_itr, BDs
    
    # SLST method
    elif method == 'SLST':
        M_inver = np.load('pure_M.npy').astype('complex128')
        shadow_ket = R_shadow_state_reconstruction(shadow, M_inver)
        shadow_rho = ket_to_rho(shadow_ket)
        infid_itr, BDs = SLST_infid(rho_standard, shadow_rho)
        return infid_itr, BDs


if __name__ == '__main__':
    
    ## calibration process ##
    cali_shots = 2000

    ## tomography process ##
    repeat = 1  
    total_photon = 2000 # number of measurements
    iteration_number = 100
    exp_run = 10

    delta_mean = 0.1
    sigma_std_list = np.linspace(0, delta_mean, 6)
    
    fids_RSLST_list = []
    fids_SLST_list = []
    
    for i in range(6):
        
        sigma_std = sigma_std_list[i]
        
        fids_RSLST = np.zeros((exp_run, 20))
        fids_SLST = np.zeros((exp_run, 20))
        for exp_num in range(exp_run):
            
            
            # deltas = np.array([0.5, ]*3)
            M_inver = calibration_and_gain_Minv(cali_shots, delta_mean, sigma_std)
            
            for state_n in range(20):
                rho_standard = np.load('standard_1q_mix_state.npy')[state_n,:,:]
                shadow_origin = shadow_result(rho_standard, total_photon, delta_mean, sigma_std)
                
                ######### robust SLST method
                infid_itr, _ = state_tomography('RSLST', M_inver)
                fids_RSLST[exp_num, state_n] = np.mean(1.0-infid_itr[:, -1])
                
                ######### SLST method
                infid_itr, _ = state_tomography('SLST', M_inver)
                fids_SLST[exp_num, state_n] = np.mean(1.0-infid_itr[:, -1])
    
        
        print('sigma: ', sigma_std)
        print('RSLST fid: %.5f' %np.mean(fids_RSLST), 'std: %5f'%np.std(np.mean(fids_RSLST, axis=1)) )
        print('SLST fid: %.5f' %np.mean(fids_SLST), 'std: %5f'%np.std(np.mean(fids_SLST, axis=1)) )
        
        fids_RSLST_list.append(fids_RSLST)
        fids_SLST_list.append(fids_SLST)
    
    # np.save('./result_data/fids_RSLST_vs_sigma_0.05.npy', fids_RSLST_list)
    # np.save('./result_data/fids_SLST_vs_sigma_0.05.npy', fids_SLST_list)
    # print('Results saved')
    time_end = time.time()
    print('Time cost = %fs' % (time_end - time_start))
