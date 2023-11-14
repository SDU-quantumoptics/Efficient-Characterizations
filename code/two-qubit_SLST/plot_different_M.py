# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, FixedLocator, NullFormatter



def load_data(state, total_photon):
    
    fids_SLST = np.load('./result_data/SLST_fidelity_state_'+str(state)+'_M='+str(total_photon)+'.npy')[:,-1]
    fids_MLE = np.load('./result_data/MLE_fidelity_state_'+str(state)+'_M='+str(total_photon)+'.npy')

    return fids_SLST, fids_MLE


def plot_fids_photons(fid1, fid2, fid3, tomo_fid1, tomo_fid2, tomo_fid3):
    x = photon_list

    std1 = np.std(fid1, axis=0, ddof = 1)
    std2 = np.std(fid2, axis=0, ddof = 1)
    std3 = np.std(fid3, axis=0, ddof = 1)
    
    mean1 = np.mean(fid1, axis=0)
    mean2 = np.mean(fid2, axis=0)
    mean3 = np.mean(fid3, axis=0)
    
    tomo_mean1 = np.mean(tomo_fid1, axis=0)
    tomo_mean2 = np.mean(tomo_fid2, axis=0)
    tomo_mean3 = np.mean(tomo_fid3, axis=0)
    
    tomo_std1 = np.std(tomo_fid1, axis=0)
    tomo_std2 = np.std(tomo_fid2, axis=0)
    tomo_std3 = np.std(tomo_fid3, axis=0)
    
    # color1, color2, color3 = tuple(np.array([101,195,234])/255), tuple(np.array([226,117,155])/255),tuple(np.array([140,167,76])/255)
    # color1, color2, color3 = tuple(np.array([140,199,181])/255), tuple(np.array([209,186,116])/255),tuple(np.array([240,150,128])/255)
    color1, color2, color3 = tuple(np.array([142, 207, 201])/255),tuple(np.array([255, 190, 122])/255),tuple(np.array([250, 127, 111])/255)

    fig = plt.figure(figsize=(1.55*1.103,1.55*1.103))
    

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    ################################################################################################################
    ax2 = plt.subplot(111)
    # ax2.set_aspect(1)

    plt.yscale('log',base=10)
    plt.xscale('log',base=2)

    # plt.fill_between(x, 1/(1-mean1-std1/2), 1/(1-mean1+std1/2), alpha = 0.2, facecolor=color1, zorder=2)
    plt.plot(x, 1/(1-mean1),marker='o',markersize=ms,ls='-', linewidth=lw, color=color1, zorder=3)
    error1 = 1/(1-mean1-std1/2)-1/(1-mean1)
    error2 = 1/(1-mean1)-1/(1-mean1+std1/2)
    plt.errorbar(x, 1/(1-mean1), yerr=np.stack([error2, error1]),linewidth=0, ecolor=color1
                  ,elinewidth=cpthick, capsize=cpsize, capthick=cpthick, barsabove=False,color=color1, zorder=5)   
    
    # plt.fill_between(x, 1/(1-mean2-std2/2), 1/(1-mean2+std2/2), alpha = 0.2, facecolor=color2, zorder=2)
    plt.plot(x, 1/(1-mean2),marker='o',markersize=ms,ls='-', linewidth=lw, color=color2, zorder=4)
    error1 = 1/(1-mean2-std2/2)-1/(1-mean2)
    error2 = 1/(1-mean2)-1/(1-mean2+std2/2)
    plt.errorbar(x, 1/(1-mean2), yerr=np.stack([error2, error1]),linewidth=0, ecolor=color2
                  ,elinewidth=cpthick, capsize=cpsize, capthick=cpthick, barsabove=False,color=color2, zorder=5) 
    
    # plt.fill_between(x, 1/(1-mean3-std3/2), 1/(1-mean3+std3/2), alpha = 0.2, facecolor=color3, zorder=2)
    plt.plot(x, 1/(1-mean3),marker='o',markersize=ms,ls='-', linewidth=lw, color=color3, zorder=5)
    error1 = 1/(1-mean3-std3/2)-1/(1-mean3)
    error2 = 1/(1-mean3)-1/(1-mean3+std3/2)
    plt.errorbar(x, 1/(1-mean3), yerr=np.stack([error2, error1]),linewidth=0, ecolor=color3
                  ,elinewidth=cpthick, capsize=cpsize, capthick=cpthick, barsabove=False,color=color3, zorder=5) 


    plt.plot(x, 1/(1-tomo_mean1),marker='s',markersize=ms,ls='-', linewidth=lw, color=color1, zorder=6)
    error1 = 1/(1-tomo_mean1-tomo_std1/2)-1/(1-tomo_mean1)
    error2 = 1/(1-tomo_mean1)-1/(1-tomo_mean1+tomo_std1/2)
    plt.errorbar(x, 1/(1-tomo_mean1), yerr=np.stack([error2, error1]),linewidth=0, ecolor=color1
                  ,elinewidth=cpthick, capsize=cpsize, capthick=cpthick, barsabove=False,color=color1, zorder=5) 
    
    
    plt.plot(x, 1/(1-tomo_mean2),marker='s',markersize=ms,ls='-', linewidth=lw, color=color2, zorder=6)
    error1 = 1/(1-tomo_mean2-tomo_std2/2)-1/(1-tomo_mean2)
    error2 = 1/(1-tomo_mean2)-1/(1-tomo_mean2+tomo_std2/2)
    plt.errorbar(x, 1/(1-tomo_mean2), yerr=np.stack([error2, error1]),linewidth=0, ecolor=color2
                  ,elinewidth=cpthick, capsize=cpsize, capthick=cpthick, barsabove=False,color=color2, zorder=5) 
    
    
    plt.plot(x, 1/(1-tomo_mean3),marker='s',markersize=ms,ls='-', linewidth=lw, color=color3, zorder=6)
    error1 = 1/(1-tomo_mean3-tomo_std3/2)-1/(1-tomo_mean3)
    error2 = 1/(1-tomo_mean3)-1/(1-tomo_mean3+tomo_std3/2)
    plt.errorbar(x, 1/(1-tomo_mean3), yerr=np.stack([error2, error1]),linewidth=0, ecolor=color3
                  ,elinewidth=cpthick, capsize=cpsize, capthick=cpthick, barsabove=False,color=color3, zorder=5) 

        
    plt.ylim(1,100)
    # plt.xlim(1,50)
    plt.yticks(ticks=[1, 10, 100], labels=['0', '0.9', '0.99'])
    plt.xticks(ticks=[50, 100, 200, 400, 800, 1600, 3200], labels=['50', '', '200', '', '800', '', '3200'])
    plt.tick_params(labelsize=7) 

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
    plt.xlabel('Used photons', fontdict={'family' : 'Arial', 'size' : 8})
    # plt.legend(labels=['SLST ()','SLST ()','SLST ()', 'MLE ()', 'MLE ()', 'MLE ()'], ncol=1, loc='lower right', prop={'family' : 'Arial', 'size'   : 7}).set_zorder(100)
    plt.grid(True)
    plt.show()
    # plt.savefig("./result_figure/fig1_different_M.pdf", format="pdf") # save figure
    return fig
    
    
    
if __name__ == '__main__':
    total_photon = 2000
    lw = 0.5
    ms = 2
    cpthick = 0.5
    cpsize = 2
    
    fids_SLST = []
    fids_tomo = []
    
    photon_list = (2**np.arange(0, 7))*50
    for i in range(3):
        fid_SLST = np.zeros((5, 7))
        fid_tomo = np.zeros((5, 7))
        state = i+1
        for tp in range(7):
            fid_SLST[:, tp], fid_tomo[:, tp] = load_data(state, photon_list[tp])
            
        fids_SLST.append(fid_SLST)
        fids_tomo.append(fid_tomo)
        
    plot_fids_photons(fids_SLST[0], fids_SLST[1], fids_SLST[2], fids_tomo[0], fids_tomo[1], fids_tomo[2])