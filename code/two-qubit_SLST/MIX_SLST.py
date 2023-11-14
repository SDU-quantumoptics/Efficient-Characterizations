# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit


# custom functions
from function_basic import rho_to_paras_mix, rho_2q_t, rho_fidelity, statistic_shadow
from func_shadow_fast import shadow_state_reconstruction
from func_kwiat_tomo import k_tomo_2q


time_start = time.time()

# # #################################################### plot fuction
def plot_infid(infids):
    fig = plt.figure()
    infid_mean = np.mean(infids, axis=0)
    infid_std = np.std(infids, axis=0, ddof = 1)
    plt.rcParams.update(
        {'text.usetex': False,'font.family': 'stixgeneral','mathtext.fontset': 'stix',})
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    itrs = np.linspace(1, iteration_number, iteration_number)
    
    plt.loglog(itrs, infid_mean)
    plt.xlabel(r'$iteration$',fontsize='large')
    plt.ylabel('$Infidelity$',fontsize='large')
    plt.tick_params(which='both', top=True,bottom=True,left=True,right=True, gridOn = True, grid_alpha = 0.3)
    plt.title('State: '+str(state), fontsize='large')
    plt.ylim(10**-2, 1)
    plt.xlim(1, iteration_number)
    plt.fill_between(itrs, infid_mean-infid_std, infid_mean+infid_std, alpha = 0.5)
    
    plt.show()     




############################################ SLST functions


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


# SLST
def SLST_infid(rho_standard, rho_shadow):
    s = 0.602
    t = 0.101
    A = 0
    Δ = 1

    # a = 12.25 # hyperparameters of bad guess 
    # b = 0.35

    a = 8.533 # hyperparameters of better guess 
    b = 1.421
    
    qubit_num = 2
    para_num = 4**qubit_num
    infids = np.ones((repeat, iteration_number))
    rho_para = np.zeros((repeat, para_num)) # with para, we can reconstruct state using mixed state model: "rho_2q_t"
    
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



    
def MLE_infid(shadow):

    counts_meta = statistic_shadow(shadow)
    tomo_rho = k_tomo_2q(counts_meta)
    fid_tomo = rho_fidelity(tomo_rho, rho_standard)
    return 1.0-fid_tomo

def state_tomography(total_photon, method):
    
    shadow = (shadow_origin[0][:total_photon], shadow_origin[1][:total_photon])
    shadow_rho = shadow_state_reconstruction(shadow[0], shadow[1])
    
        # SLST method
    if method == 'SLST':
        infid_itr, BDs = SLST_infid(rho_standard, shadow_rho)
        return infid_itr, BDs
    
        # MLE method
    elif method == 'MLE':
        infid_MLE = MLE_infid(shadow)
        return np.mean(1.0-infid_MLE)
    

if __name__ == '__main__':
    repeat = 2
    photon_list = np.array([  50,  100,  200,  400,  800, 1600, 3200])
    iteration_number = 200
    
    state = 1 # state1 eta=0.37, state2 eta=0.06, state3 eta=0.87
    nn = 5 # choose how many photons are used for tomography
    total_photon = photon_list[nn]


    fids = []
    infids = []
    state_paras = []
    MLE_fids = []
    
    for exp_num in range(5):
        rho_standard = np.load('standard_rho'+str(state)+'.npy')
        shadow_origin = np.load("./exp_shadow/state"+str(state)+"/shadow_M="+
                                str(total_photon)+"_"+str(exp_num)+".npy")
        shadow_origin = (shadow_origin[0], shadow_origin[1].astype('int64'))
        
        # MLE
        MLE_fid = state_tomography(total_photon, method='MLE')
        MLE_fids.append(MLE_fid)
            
        # SLST
        infid_itr, state_para = state_tomography(total_photon, method='SLST')
        infids.append(np.mean(infid_itr, axis=0)) # for plot
        fids.append(np.mean(1.0-infid_itr[:, -1]))
        state_paras.append(state_para)
        
    print('SLST fid: %.5f' %np.mean(fids))
    print('MLE fid: %.5f' %np.mean(MLE_fids))
    
    # np.save('./result_data/SLST_fidelity_state_'+str(state)+'_M='+str(total_photon)+'.npy', 1.0-np.array(infids))
    # np.save('./result_data/SLST_iterative_prameters_state_'+str(state)+'_M='+str(total_photon)+'.npy', state_paras)
    # np.save('./result_data/MLE_fidelity_state_'+str(state)+'_M='+str(total_photon)+'.npy', MLE_fids)
    # print('Results saved')
    # plot_infid(infids)


    
    time_end = time.time()
    print('Time cost = %fs' % (time_end - time_start))

