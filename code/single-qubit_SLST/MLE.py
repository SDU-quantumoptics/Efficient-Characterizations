# -*- coding: utf-8 -*-

import numpy as np
from func_basic import rho_fidelity, shadow_to_tomo_inpt
from func_kwiat_tomo import k_tomo




photon_list = [15, 30, 60, 120, 240, 315]
state_number = 20
ST = np.load('standard_20states.npy')

nn = 5 # choose how many photons are used from photon_list for tomography

MLE_fids = np.zeros((state_number, 10))

for sn in range(state_number):
    ket = ST[sn, :, :]
    rho_standard = ket @ ket.conj().T
    
    for exp_num in range(10):
        CST = np.load('./exp_shadow/state'+str(sn)+'/shadow_'+'M='+str(photon_list[nn])+'_'+str(exp_num)+'.npy')
        shadow = (CST[0], CST[1].astype('int32'))

        
        tomo_input = shadow_to_tomo_inpt(shadow)
        MLE_rho = k_tomo(tomo_input)
        MLE_fids[sn, exp_num] = rho_fidelity(MLE_rho, rho_standard)

print('mean fidelity (MLE):', np.mean(MLE_fids))
