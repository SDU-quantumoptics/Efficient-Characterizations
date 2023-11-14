# -*- coding: utf-8 -*-

import numpy as np
# import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter




def plot_func(var_x_ls):
    
    x = photon_ls
    var_x_max = np.max(var_x_ls, axis=1)
    var_x_max_mean = np.mean(var_x_max, axis=1) 
    var_x_max_std = np.std(var_x_max, axis=1)
    
    color1, color2 = tuple(np.array([162, 20, 47])/255), tuple(np.array([34, 140, 141])/255)
    fig = plt.figure(figsize=(4.7,3.5))
    ax2 = plt.subplot(111)
    
    # plot
    plt.plot(x, var_x_max_mean,marker='o',markersize=5,ls='-', linewidth=1, color=color1, zorder=3)
    plt.errorbar(x, var_x_max_mean, yerr=var_x_max_std, ecolor='black', linewidth=0
                  ,elinewidth=1, capsize=2, capthick=0.5, barsabove=False,color='black', zorder=3)
    plt.axhline(y = 0.75, color =color2, linestyle ="--")
    
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.ylim(0.4,1.2)
    # plt.xlim(1,50)
    plt.yticks(ticks=[0.4, 0.75,1, 1.2])
    # plt.xticks(ticks=x, labels=x.astype('str'))
    plt.tick_params(labelsize=11) 
    
    x1_label = ax2.get_xticklabels() 
    [x1_label_temp.set_fontname('Arial') for x1_label_temp in x1_label]
    y1_label = ax2.get_yticklabels() 
    [y1_label_temp.set_fontname('Arial') for y1_label_temp in y1_label]
    ax2.yaxis.set_minor_locator(FixedLocator(10/np.concatenate((np.arange(1,10,1), np.arange(0.1,1,0.1), np.arange(0.01,0.1,0.01)))))
    ax2.yaxis.set_minor_formatter(NullFormatter())

    bwith = 1
    ax2.spines['top'].set_linewidth(bwith)
    ax2.spines['right'].set_linewidth(bwith)
    ax2.spines['bottom'].set_linewidth(bwith)
    ax2.spines['left'].set_linewidth(bwith)
    ax2.spines['top'].set_linewidth(bwith)
    ax2.spines['right'].set_linewidth(bwith)

    plt.tick_params(which='both', top=True,bottom=True,left=True,right=True, gridOn = True, grid_alpha = 0.0, width=0.5, size=1, grid_linewidth=0.5)
    
    plt.ylabel('Variance', fontdict={'family' : 'Arial', 'size' : 14}) 
    plt.xlabel('Photons', fontdict={'family' : 'Arial', 'size' : 14})
    plt.legend(labels=['Octahedron POVM'], ncol=1, loc='lower right', prop={'family' : 'Arial', 'size'   : 12},
               frameon=False).set_zorder(100)
    plt.grid(True)
    plt.show()
    return fig


###############################################################################################
H = np.array([[1], [0]], dtype='complex128')
V = np.array([[0], [1]], dtype='complex128')
A = (H+V)/np.sqrt(2)
D = (H-V)/np.sqrt(2)
R = (H+1j*V)/np.sqrt(2)
L = (H-1j*V)/np.sqrt(2)
Identity = np.eye(2)
E_k = np.array([H @ H.conj().T, V @ V.conj().T, A @ A.conj().T, D @ D.conj().T, R @ R.conj().T, L @ L.conj().T])/3
rho_k = 9*E_k - np.repeat(Identity[np.newaxis, :, :], 6, axis=0)




def var_exp_define(X, shadow):
    shadow_counts = len(shadow[0])
    o_hat = np.zeros(shadow_counts, dtype='float64')
    k_index = np.array([[3, 2], 
                        [5, 4],
                        [1, 0]])
    
    for i in range(shadow_counts):
        index = k_index[int(shadow[1][i]), int((shadow[0][i]+1)/2)]
        o_i = np.trace(X @ rho_k[index, :, :])
        # print(o_i)
        o_hat[i] = np.real(o_i)
        
    variance = np.mean((o_hat-np.mean(o_hat))**2)
    return variance


if __name__ == '__main__':
    
        
    observable = np.array([[1], [1]], dtype='complex128')/np.sqrt(2) # this indicates the observable |+><+|
    X = observable @ observable.conj().T 

    standard_states = np.load('standard_20states.npy')
    shadow_outcome = np.load('shadow_outcome.npy')
    shadow_pauli = np.load('shadow_pauli.npy')
    
    photon_ls = np.linspace(15, 315, 21, dtype='int32') 
    variance_exp_ls = np.zeros((len(photon_ls), 20, 10), dtype='float64') # 20 states, 10 repeats

    for photon_index in range(len(photon_ls)):
        total_photon = photon_ls[photon_index]
        rand_index = np.random.choice(315, total_photon, replace=False)-1
    
        for state_n in range(20): 
            state_ket = standard_states[state_n]
            state_rho = state_ket @ state_ket.conj().T
        
            for num in range(10):
                shadow = (shadow_outcome[state_n, num, rand_index, :], 
                          shadow_pauli[state_n, num, rand_index, :])
                variance_exp = var_exp_define(X, shadow)
                variance_exp_ls[photon_index, state_n, num] = np.real(variance_exp) 
    
        print('Variance for 20 projectors:', np.mean(np.real(variance_exp_ls[photon_index]), axis=1))
    
    
    plot_func(variance_exp_ls) 
    # np.save('variance_exp_x_M_errorbar.npy', variance_exp_ls)
    
    
