# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, FixedLocator, NullFormatter


def load_data(state, j):
    fids_RSLST = np.load('./result_data/noise_RSLST_fidelity_state_'+str(state)+'_j='+str(j)+'.npy')[:, -1]
    fids_MLE = np.load('./result_data/noise_MLE_fidelity_state_'+str(state)+'_j='+str(j)+'.npy')
    fids_SLST = np.load('./result_data/noise_SLST_fidelity_state_'+str(state)+'_j='+str(j)+'.npy')[:, -1]

    return fids_RSLST, fids_MLE, fids_SLST


def plot_fids_noises(final_denoise, final_tomo, final_normal):
    x = np.arange(1, 5+1, 1)

    # the uncertainty of Robust SLST considering both calibration and tomography is simulated by Monte Carlo method
    std_de = np.load('./simul_calibration_shot_noise/uncertainty_cali_state_'+str(state)+'.npy') # load simulation results
    
    std_to = np.std(final_tomo, axis=1, ddof = 1)
    std_no = np.std(final_normal, axis=1, ddof = 1)

    
    mean_de = np.mean(final_denoise, axis=1)
    mean_to = np.mean(final_tomo, axis=1)
    mean_no = np.mean(final_normal, axis=1)

    
    # color1, color2, color3 = tuple(np.array([142,207,201])/255), tuple(np.array([255,190,122])/255),tuple(np.array([245,130,106])/255)
    # color4, color5 = tuple(np.array([130,176,210])/255), tuple(np.array([190,184,220])/255)
    color1, color2, color3 = tuple(np.array([252, 141, 98])/255),tuple(np.array([102, 194, 165])/255),tuple(np.array([141, 160, 203])/255)
    
    fig = plt.figure(figsize=(2.07*1.104,1.55*1.104)) #2.54cm =1 inch
    

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    ################################################################################################################
    ax2 = plt.subplot(111)

    plt.yscale('log',base=10)
    # plt.xscale('log',base=2)


    plt.plot(x, 1/(1-mean_de),marker='D',markersize=ms,linewidth=lw, color=color2, zorder=3)
    error1 = 1/(1-mean_de-std_de/2)-1/(1-mean_de)
    error2 = 1/(1-mean_de)-1/(1-mean_de+std_de/2)
    plt.errorbar(x, 1/(1-mean_de), yerr=np.stack([error2, error1]),linewidth=0, ecolor=color2
                  ,elinewidth=cpthick, capsize=cpsize, capthick=cpthick, barsabove=False,color=color2, zorder=5) 
    
    
    plt.plot(x, 1/(1-mean_no),marker='o',markersize=ms,linewidth=lw, color=color3, zorder=5)
    error1 = 1/(1-mean_no-std_no/2)-1/(1-mean_no)
    error2 = 1/(1-mean_no)-1/(1-mean_no+std_no/2)
    plt.errorbar(x, 1/(1-mean_no), yerr=np.stack([error2, error1]),linewidth=0, ecolor=color3
                  ,elinewidth=cpthick, capsize=cpsize, capthick=cpthick, barsabove=False,color=color3, zorder=5)   
    
    
    plt.plot(x, 1/(1-mean_to),marker='s',markersize=ms,linewidth=lw, color=color1, zorder=4)
    error1 = 1/(1-mean_to-std_to/2)-1/(1-mean_to)
    error2 = 1/(1-mean_to)-1/(1-mean_to+std_to/2)
    plt.errorbar(x, 1/(1-mean_to), yerr=np.stack([error2, error1]),linewidth=0, ecolor=color1
                  ,elinewidth=cpthick, capsize=cpsize, capthick=cpthick, barsabove=False,color=color1, zorder=5)   
    
  

        
    plt.ylim(1,100)
    # plt.xlim(1,50)
    plt.yticks(ticks=[1, 10, 100], labels=['0', '0.9', '0.99'])
    plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['0', '1.5', '3.5', '5.8', '8.6'])
    plt.tick_params(labelsize=7) 

    x1_label = ax2.get_xticklabels() 
    [x1_label_temp.set_fontname('Arial') for x1_label_temp in x1_label]
    y1_label = ax2.get_yticklabels() 
    [y1_label_temp.set_fontname('Arial') for y1_label_temp in y1_label]
    ax2.yaxis.set_minor_locator(FixedLocator(10/np.concatenate((np.arange(1,10,1), np.arange(0.1,1,0.1), np.arange(0.01,0.1,0.01)))))
    ax2.yaxis.set_minor_formatter(NullFormatter())


    #  db = [1.45, 3.54, 5.82, 8.61]
    bwith = 0.5
    ax2.spines['top'].set_linewidth(bwith)  
    ax2.spines['right'].set_linewidth(bwith) 
    ax2.spines['bottom'].set_linewidth(bwith)
    ax2.spines['left'].set_linewidth(bwith)
    ax2.spines['top'].set_linewidth(bwith)
    ax2.spines['right'].set_linewidth(bwith)

    plt.tick_params(which='both', top=True,bottom=True,left=True,right=True, gridOn = True, grid_alpha = 0.0, width=0.5, size=1, grid_linewidth=0.5)
    
    plt.ylabel('Fidelity', fontdict={'family' : 'Arial', 'size' : 8}) 
    plt.xlabel('Noise (dB)', fontdict={'family' : 'Arial', 'size' : 8})
    # plt.legend(labels=['Robust SLST','SLST','MLE'], ncol=1, loc='lower left', prop={'family' : 'Arial', 'size'   : 7}).set_zorder(100)
    
    plt.grid(True)
    # plt.title('State '+str(state), fontdict={'family' : 'Arial', 'size' : 7})
    plt.show()
    # plt.savefig("./result_figure/fig3_noise_state"+str(state)+".pdf", format="pdf") # save figure
    return fig
    
    
    
if __name__ == '__main__':
    total_photon = 1000
    lw = 0.5
    ms = 2
    cpthick = 0.5
    cpsize = 2
    
    fids_RSLST = []
    fids_MLE = []
    fids_SLST = []
    state = 1
    for j in range(5):
        
        fid_RSLST, fid_MLE, fid_SLST = load_data(state, j)
        fids_RSLST.append(fid_RSLST)
        fids_MLE.append(fid_MLE)
        fids_SLST.append(fid_SLST)
        
    # plot the results of fidelity using 3 methods
    plot_fids_noises(fids_RSLST, fids_MLE, fids_SLST)
        
    
    
    

    