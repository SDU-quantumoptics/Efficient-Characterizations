# -*- coding: utf-8 -*-

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
from func_1shadow import shadow_state_reconstruction


def plot_func(var_max):
    
    x = np.arange(1, 128+1, 1)
    color1, color2 = tuple(np.array([162, 20, 47])/255), tuple(np.array([0, 114, 189])/255)
    fig, ax = plt.subplots()
    
    plt.plot(x, var_max, marker='o',markersize=5,ls='None', linewidth=2, color=color1, zorder=3)
    plt.axhline(y = 1.5/2, color =color2, linestyle ="--")
    
    
    plt.ylim(0,1)
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.tick_params(labelsize=18)
    x1_label = ax.get_xticklabels() 
    [x1_label_temp.set_fontname('Arial') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels() 
    [y1_label_temp.set_fontname('Arial') for y1_label_temp in y1_label]
    ax.yaxis.set_minor_formatter(NullFormatter())
    bwith = 1
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    # plt.tick_params(which='both', top=True,bottom=True,left=True,right=True, gridOn = True, grid_alpha = 0.0, width=0.5, size=1, grid_linewidth=0.5)
    
    plt.ylabel('max variance', fontdict={'family' : 'Arial', 'size' : 20}) 
    plt.xlabel('Observables', fontdict={'family' : 'Arial', 'size' : 20})
    plt.show()
    # plt.savefig("norm_exp.pdf", format="pdf")



############################################################################################### SIC POVM elements
H = np.array([[1], [0]], dtype='complex128')
V = np.array([[0], [1]], dtype='complex128')
A = (H+V)/np.sqrt(2)
D = (H-V)/np.sqrt(2)
R = (H+1j*V)/np.sqrt(2)
L = (H-1j*V)/np.sqrt(2)
Identity = np.eye(2)
E_k = np.array([H @ H.conj().T, V @ V.conj().T, A @ A.conj().T, D @ D.conj().T, R @ R.conj().T, L @ L.conj().T])/3
rho_k = 9*E_k - np.repeat(Identity[np.newaxis, :, :], 6, axis=0)


sic1 = H
sic2 = 1/np.sqrt(3)*H+np.sqrt(2/3)*V
sic3 = 1/np.sqrt(3)*H+np.sqrt(2/3)*np.exp(2*np.pi*1j/3)*V
sic4 = 1/np.sqrt(3)*H+np.sqrt(2/3)*np.exp(-2*np.pi*1j/3)*V

E_k_sic = np.array([sic1 @ sic1.conj().T, sic2 @ sic2.conj().T, sic3 @ sic3.conj().T, sic4 @ sic4.conj().T])/2
rho_k_sic = 6*E_k_sic - np.repeat(Identity[np.newaxis, :, :], 4, axis=0)

#################################################################################################
# @jit(nopython=True)
def var_exp_define(X, outcomes, paulis):
    shadow_counts = len(outcomes)
    o_hat = np.zeros(shadow_counts, dtype='float64')
    k_index = np.array([[3, 2], 
                        [5, 4],
                        [1, 0]], dtype='int32')
    
    for i in range(shadow_counts):
        index = k_index[int(shadow[1][i]), int((shadow[0][i]+1)/2)]
        o_i = np.trace(X @ rho_k[index, :, :])
        # print(o_i)
        o_hat[i] = np.real(o_i)
        
    variance = np.mean((o_hat-np.mean(o_hat))**2)
    return variance


def var_sic(X, rho):
    var_k = []
    for k in range(4):
        v_k = np.trace(rho_k_sic[k] @ X)**2 * np.trace(rho @ E_k_sic[k])
        var_k.append(v_k)
    return np.sum(var_k) - np.trace(rho @ X)**2

if __name__ == '__main__':
    
    shadow_outcome = np.load('shadow_outcome.npy')
    shadow_pauli = np.load('shadow_pauli.npy')
    
    variance_exp_ls = np.zeros((128, 20, 10), dtype='float64')
    variance_sic_ls = np.zeros((128, 20, 10), dtype='float64') 

    total_photon = 315
    rand_index = np.random.choice(315, total_photon, replace=False)-1

    for X_n in range(128): # generate 128 observables randomly
        X_ket = np.array(qt.rand_ket_haar(2))
        X = X_ket @ X_ket.conj().T 
        
        for state_n in range(20):
        
            for num in range(10):
                shadow = (shadow_outcome[state_n, num, rand_index, :], 
                          shadow_pauli[state_n, num, rand_index, :])
                variance_exp = var_exp_define(X, shadow[0], shadow[1]) 
                variance_exp_ls[X_n, state_n, num] = np.real(variance_exp) 
                
                shadow_state = shadow_state_reconstruction(shadow)
                variance_sic = var_sic(X, shadow_state)
                variance_sic_ls[X_n, state_n, num] = np.real(variance_sic)
    
        print(X_n)
    
    # np.save('variance_exp_ls_315pho.npy', variance_exp_ls)
    # np.save('variance_sic_ls_315pho.npy', variance_sic_ls)
    
    var_mean = np.mean(variance_exp_ls, axis=2)
    var_max = np.max(var_mean, axis=1)
    plot_func(var_max) 
    
    var_mean = np.mean(variance_sic_ls, axis=2)
    var_max = np.max(var_mean, axis=1)
    plot_func(var_max) 

