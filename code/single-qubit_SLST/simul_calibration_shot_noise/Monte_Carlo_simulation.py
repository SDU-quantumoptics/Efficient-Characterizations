# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit


# custom functions
from function_basic import rho_to_paras_mix, rho_2q_t, rho_fidelity
from function_robust_shadow import R_shadow_state_reconstruction, ket_to_rho
from function_sample_channelM import calibration_and_gain_Minv


time_start = time.time()

############################################ start iteration


@jit(nopython=True)
def coff_delta(BD1, BD2, delta):
    qubit_num = 2
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

    a = 8.533 # better guess hyperparameters
    b = 1.421
    
    qubit_num = 2
    para_num = 4**qubit_num
    infids = np.ones((repeat, iteration_number))
    rho_para = np.zeros((repeat, para_num)) # with para, we can reconstruct state using mixed state model
    
    for i in range(repeat):
        # 2.set a measurement basis B
        # B_coff = [1,]*para_num
        # B_coff = np.array(B_coff, dtype='float64') # use this: bad guess
        B_coff = rho_to_paras_mix(rho_shadow.copy(), qubit_num=2)+0.2 # use this: good guess
        
        for itr in range(iteration_number):

            k = itr+50
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


    


def state_tomography(total_photon, method, M_inver):
    
    shadow = (shadow_origin[0][:total_photon], shadow_origin[1][:total_photon])
    
    # robust SLST method
    if method == 'RSLST':
        shadow_ket = R_shadow_state_reconstruction(shadow, M_inver)
        shadow_rho = ket_to_rho(shadow_ket)
        infid_itr, BDs = SLST_infid(rho_standard, shadow_rho)
        return infid_itr, BDs
    
    # # SLST method
    # elif method == 'SLST':
    #     shadow_rho = shadow_state_reconstruction(shadow[0], shadow[1])
    #     infid_itr, BDs = SLST_infid(rho_standard, shadow_rho)
    #     return infid_itr, BDs
    

    


if __name__ == '__main__':
    
    repeat = 2  
    total_photon = 1000 # number of measurements
    iteration_number = 200
    
    state = 1   # state1 eta=0.37, state2 eta=0.06, state3 eta=0.87
    j = 0       # noise level j=0: 0dB, j=1: 1.5dB, j=2: 3.5dB, j=3: 5.8dB, j=4: 8.6dB
    cali_probs = np.load('cali_probs_j='+str(j)+'.npy')

    
    MLE_fids = []

    fids_RSLST= []
    infids_RSLST = []
    state_paras_RSLST = []


    for exp_num in range(5):
        rho_standard = np.load('standard_rho'+str(state)+'.npy')
        shadow_origin = np.load("../exp_shadow/noise_state"+str(state)+"/shadow_M=1000_noise_"+str(j)+"_"+str(exp_num)+".npy")
        shadow_origin = (shadow_origin[0], shadow_origin[1].astype('int64'))
        
        
        M_inver = calibration_and_gain_Minv(cali_probs)
        ######### robust SLST method
        infid_itr, state_para = state_tomography(total_photon, 'RSLST', M_inver)
        infids_RSLST.append(np.mean(infid_itr, axis=0)) # for plot
        fids_RSLST.append(np.mean(1.0-infid_itr[:, -1]))
        state_paras_RSLST.append(state_para)
        
        
        
    
    # used for calculating standard deviation considering shot noise of calibration
    # np.save('./result_data/noise_RSLST_fidelity_state_'+str(state)+'_j='+str(j)+'.npy', 1.0-np.array(infids_RSLST)) 
    # print('Monte Carlo results saved')
    
    time_end = time.time()
    print('Time cost = %fs' % (time_end - time_start))
    
    