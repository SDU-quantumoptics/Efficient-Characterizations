# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, FixedLocator, NullFormatter


def load_data(state, j=0):
    
    fids_SLST = np.load('./result_data(itr)/SLST_fidelity_state_'+str(state)+'_M='+str(2000)+'.npy')
    fids_MLE = np.load('./result_data(itr)/MLE_fidelity_state_'+str(state)+'_M='+str(2000)+'.npy')

    return fids_SLST, fids_MLE



def plot_fids_k(fid1, fid2, fid3, tomo_fids):
    x_index = np.array([  1,   3,   6,  10,  16,  22,  34,  52,  76, 120, 144, 200])-1
    x = np.arange(1, iteration_number+1, 1)
    xnew = x[x_index]

    std1 = np.std(fid1[:, x_index], axis=0, ddof = 1)
    std2 = np.std(fid2[:, x_index], axis=0, ddof = 1)
    std3 = np.std(fid3[:, x_index], axis=0, ddof = 1)
    
    mean1 = np.mean(fid1[:, x_index], axis=0)
    mean2 = np.mean(fid2[:, x_index], axis=0)
    mean3 = np.mean(fid3[:, x_index], axis=0)
    
    # color1, color2, color3 = tuple(np.array([76,143,186])/255), tuple(np.array([250,132,110])/255),tuple(np.array([114,90,124])/255)
    # color4 = tuple(np.array([197,109,133])/255)
    # colora, colorb = color1, color2
    # color1, color2, color3 = tuple(np.array([140,199,181])/255), tuple(np.array([209,186,116])/255),tuple(np.array([240,150,128])/255)
    color1, color2, color3 = tuple(np.array([142, 207, 201])/255),tuple(np.array([255, 190, 122])/255),tuple(np.array([250, 127, 111])/255)
    # color1, color2, color3 = tuple(np.array([206, 70, 45])/255),tuple(np.array([142, 207, 201])/255),tuple(np.array([169, 73, 109])/255)
    
    fig = plt.figure(figsize=(1.55*1.103,1.55*1.103))
    

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    ################################################################################################################2
    ax2 = plt.subplot(111)
    # ax2.set_aspect(3)
    
    plt.yscale('log',base=10)
    plt.xscale('log',base=2)

    plt.fill_between(xnew, 1/(1-mean1-std1/2), 1/(1-mean1+std1/2), alpha = 0.3, facecolor=color1, zorder=2)
    plt.plot(xnew, 1/(1-mean1),marker='None',markersize=16,ls='--', linewidth=lw1, color=color1, zorder=3)
    
    plt.fill_between(xnew, 1/(1-mean2-std2/2), 1/(1-mean2+std2/2), alpha = 0.3, facecolor=color2, zorder=2)
    plt.plot(xnew, 1/(1-mean2),marker='None',markersize=16,ls='--', linewidth=lw1, color=color2, zorder=4)
    
    plt.fill_between(xnew, 1/(1-mean3-std3/2), 1/(1-mean3+std3/2), alpha = 0.3, facecolor=color3, zorder=2)
    plt.plot(xnew, 1/(1-mean3),marker='None',markersize=16,ls='--', linewidth=lw1, color=color3, zorder=5)

    plt.axhline(1/(1-np.mean(tomo_fids[0])), linewidth=lw1, color=color1, zorder=6)
    plt.axhline(1/(1-np.mean(tomo_fids[1])), linewidth=lw1, color=color2, zorder=6)
    plt.axhline(1/(1-np.mean(tomo_fids[2])), linewidth=lw1, color=color3, zorder=6)
    
        
    plt.ylim(1,100)
    plt.xlim(1,200)
    plt.yticks(ticks=[1, 10, 100], labels=['0', '0.9', '0.99'])
    plt.xticks(ticks=[  1,   4,  16,  64, 200], labels=['1', '4', '16' ,'64', '200'])
    plt.tick_params(labelsize=pt8) 

    x1_label = ax2.get_xticklabels() 
    [x1_label_temp.set_fontname('Arial') for x1_label_temp in x1_label]
    y1_label = ax2.get_yticklabels() 
    [y1_label_temp.set_fontname('Arial') for y1_label_temp in y1_label]

    ax2.yaxis.set_minor_locator(FixedLocator(10/np.concatenate((np.arange(1,10,1), np.arange(0.1,1,0.1), np.arange(0.01,0.1,0.01)))))
    ax2.yaxis.set_minor_formatter(NullFormatter())

    bwith = 0.5
    ax2.spines['top'].set_linewidth(bwith) 
    ax2.spines['right'].set_linewidth(bwith) 
    ax2.spines['bottom'].set_linewidth(bwith)
    ax2.spines['left'].set_linewidth(bwith)
    ax2.spines['top'].set_linewidth(bwith)
    ax2.spines['right'].set_linewidth(bwith)

    plt.tick_params(which='both', top=True,bottom=True,left=True,right=True, gridOn = True, grid_alpha = 0.0, width=0.5, size=1, grid_linewidth=0.5)
    
    plt.ylabel('Fidelity', fontdict={'family' : 'Arial', 'size' : 8}) 
    plt.xlabel('k', fontdict={'family' : 'Arial', 'size' : 8})
    # plt.legend(labels=['SLST ()','SLST ()','SLST ()', 'MLE ()', 'MLE ()', 'MLE ()'], ncol=1, loc='upper left', prop={'family' : 'Arial', 'size'   : 7}).set_zorder(100)
    plt.grid(True)
    plt.show()
    # plt.savefig("./result_figure/fig2_different_k.pdf", format="pdf") # save figure
    return fig
    
    
if __name__ == '__main__':
    total_photon = 2000
    iteration_number = 200
    pt10 = 8
    pt8 = 7
    lw1 = 0.5
    
    fids_SLST = []
    fids_tomo = []
    for i in range(3):
        state = i+1
        fid_SLST, fid_tomo = load_data(state)
        fids_SLST.append(fid_SLST)
        fids_tomo.append(fid_tomo)
        
    plot_fids_k(fids_SLST[0], fids_SLST[1], fids_SLST[2], fids_tomo)
        
        
    
    
    
    
    
    
    