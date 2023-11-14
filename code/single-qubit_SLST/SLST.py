# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
from func_basic import rho_fidelity, rho_nq, proj_ob
from numba import jit
from func_shadow_fast import shadow_state_reconstruction


time_start = time.time()


# # #################################################### plot fuction
def plot_infid(infids):
    plt.figure()
    infid_mean = np.mean(infids, axis=0)
    infid_std = np.std(infids, axis=0, ddof = 1)
    plt.rcParams.update(
        {'text.usetex': False,'font.family': 'stixgeneral','mathtext.fontset': 'stix',})
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    itrs = np.linspace(1, iteration_number, iteration_number)
    
    plt.yscale("log")
    plt.plot(itrs, infid_mean)
    plt.xlabel(r'$iteration$',fontsize='large')
    plt.ylabel('$Infidelity$',fontsize='large')
    plt.tick_params(which='both', top=True,bottom=True,left=True,right=True, gridOn = True, grid_alpha = 0.3)
    # plt.title('State: '+str(state), fontsize='large')
    plt.ylim(10**-2, 1)
    plt.xlim(1, iteration_number)
    plt.fill_between(itrs, infid_mean-infid_std, infid_mean+infid_std, alpha = 0.5)
    
    plt.show()




@jit(nopython=True)
def coff_delta(BD1, BD2, delta):
    para_num = 2 * 2**num_qubits - 1

    deltas = np.zeros(para_num, dtype='float64')
    
    dw = np.ones(para_num)
    option = np.array([-delta, delta], dtype='float64')
    deltas = np.random.choice(option, para_num)*dw

    BD1 += deltas
    BD2 -= deltas
        
    return BD1, BD2, deltas






def SLST_infid(rho_standard, rho_shadow):

    ################################
    s = 0.602/2
    t = 0.101
    A = 0
    Δ = 1
    
    a = 13
    b = 0.5
    ################################

    para_num = 2 * 2**num_qubits - 1
    infids = np.ones((repeat, iteration_number))
    
    rho_para = np.zeros((repeat, para_num))
    for i in range(repeat):
        # 2.set a measurement basis B
        B_coff = [1,]*para_num
        # B_coff = [random.uniform(0,1)for i in range(2)]+[random.uniform(0,4*np.pi)for i in range(1)]
        B_coff = np.array(B_coff, dtype='float64')
        
        
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
            ob1 = rho_nq(BD1_para, para_num)
            ob2 = rho_nq(BD2_para, para_num)
            
            f1 = proj_ob(ob1, rho_shadow)
            f2 = proj_ob(ob2, rho_shadow)
            

            p1 = f1
            p2 = f2

            # 5.calculate gradient gk
            αk = a/((k+1+A)**s)
            gk = (p1-p2)*Δ/(2*βk)
    
            # 6.set B2 = B1+αk*gk
            B_coff = B_coff + BD_paras[2]*gk*αk
            
            # 7.print data
            rho = rho_nq(B_coff, para_num)
            fid = rho_fidelity(rho, rho_standard)
            
            infids[i, itr] = 1.0-fid
            rho_para[i, :] = B_coff
        # print('%.5f' %fid, '\t {:.2%}'.format((i+1)/repeat))
    return np.mean(infids, axis=0), rho_para
    
    


if __name__ == '__main__':
    num_qubits = 1
    photon_list = [15, 30, 60, 120, 240, 315] # number of measurements
    repeat = 2 # repeat of SLST
    iteration_number = 30 # iteration of SLST
    state_number = 20
    
    ST = np.load('standard_20states.npy') # load 20 ideal pure states
    
    fids = np.zeros((state_number, iteration_number, 10))
    nn = 0 # choose how many photons are used from photon_list 0:M=15, 1:M=30, 2:M=60 ... 
    
    for sn in range(state_number):
        ket = ST[sn, :, :]
        rho_standard = ket @ ket.conj().T
        
        for exp_num in range(10):
            CST = np.load('./exp_shadow/state'+str(sn)+'/shadow_'+'M='+str(photon_list[nn])+'_'+str(exp_num)+'.npy')
            shadow = (CST[0], CST[1].astype('int32'))
            rho_shadow = shadow_state_reconstruction(shadow[0], shadow[1])
    
            infid = SLST_infid(rho_standard, rho_shadow)[0]
            fids[sn, :, exp_num] = 1-infid
    plot_infid(1-np.mean(fids, axis=2))

    print('Fidelity (SLST):', np.mean(fids[:, -1, :]))
    
    time_end = time.time()
    print('Time cost = %.3fs' % (time_end - time_start))