# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, FixedLocator, NullFormatter


def load_data(state, j):
    fids_RSLST_std = np.load('./result_data/noise_RSLST_fidelity_state_'+str(state)+'_j='+str(j)+'.npy')[:, -1]
    return fids_RSLST_std


    
if __name__ == '__main__':


    # calculate standard deviation (uncertainty of RSLST) according to Monte Carlo results
    fids_RSLST_std = []
    state = 3
    for j in range(5):
        
        fid_RSLST_std = load_data(state, j)
        fids_RSLST_std.append(fid_RSLST_std)
        std_cali = np.std(fids_RSLST_std, axis=1, ddof = 1)
    
    np.save('uncertainty_cali_state_'+str(state)+'.npy', std_cali)
        
    
    
    
    
    
    
    