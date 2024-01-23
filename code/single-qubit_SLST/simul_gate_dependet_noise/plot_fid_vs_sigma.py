# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, FixedLocator, NullFormatter

def plot(fid_SLST, fid_RSLST):
    
    if sigma==0.05:
        x = np.arange(0, 0.06, 0.01)
    else:
        x = np.arange(0, 0.12, 0.02)
    
    std_RSLST = np.std(fid_RSLST, axis=1)
    m_RSLST = np.mean(fid_RSLST, axis=1)
    std_SLST = np.std(fid_SLST, axis=1)
    m_SLST = np.mean(fid_SLST, axis=1)
    
    # color1, color2, color3 = tuple(np.array([76,143,186])/255), tuple(np.array([250,132,110])/255),tuple(np.array([114,90,124])/255)
    # color1, color2, color3 = tuple(np.array([114,193,191])/255), tuple(np.array([250,132,110])/255),tuple(np.array([114,90,124])/255)
    # color4 = tuple(np.array([197,109,133])/255)
    color1, color2, color3, color4 = [tuple(np.array([253, 184, 109])/255),
                                      tuple(np.array([218+10, 56+40, 43+20])/255),
tuple(np.array([56, 81, 163])/255),
tuple(np.array([146, 198, 222])/255)]
    colora, colorb = color1, color2
    
    fig = plt.figure(figsize=(3,2))
    

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    ################################################################################################################
    ax2 = plt.subplot(111)

    
    plt.yscale('log',base=10)
    # plt.xscale('log',base=1.5849)
    error1 = 1/(1-m_RSLST-std_RSLST)-1/(1-m_RSLST)
    error2 = 1/(1-m_RSLST)-1/(1-m_RSLST+std_RSLST)
    plt.plot(x, 1/(1-m_RSLST),marker='o',markersize=ms+0.5,ls='-', linewidth=lw, color=colorb, zorder=4)
    plt.errorbar(x, 1/(1-m_RSLST), yerr=np.stack([error2, error1]),linewidth=0, ecolor=colorb
                   ,elinewidth=cpthick, capsize=cpsize, capthick=cpthick, barsabove=False,color=color4, zorder=5)   
    
    error1 = 1/(1-m_SLST-std_SLST)-1/(1-m_SLST)
    error2 = 1/(1-m_SLST)-1/(1-m_SLST+std_SLST)
    plt.plot(x, 1/(1-m_SLST),marker='s',markersize=ms+0.2,ls='-', linewidth=lw, color=color4, zorder=2)
    plt.errorbar(x, 1/(1-m_SLST), yerr=np.stack([error2, error1]),linewidth=0, ecolor=color4
                   ,elinewidth=cpthick, capsize=cpsize, capthick=cpthick, barsabove=False,color=color4, zorder=3)       
    
    plt.ylim(1,1000)
    # plt.xlim(0, x[-1]*1.05)
    plt.yticks(ticks=[1, 10, 100, 1000], labels=['0', '0.9', '0.99', '0.999'])
    # plt.xticks(ticks=[30,100, 300,1000, 3000], labels=['30','100', '300','1000', '3000'])
    plt.tick_params(labelsize=7) #刻度字体大小20
     # 2.3 坐标轴刻度字体设置
    x1_label = ax2.get_xticklabels() 
    [x1_label_temp.set_fontname('Arial') for x1_label_temp in x1_label]
    y1_label = ax2.get_yticklabels() 
    [y1_label_temp.set_fontname('Arial') for y1_label_temp in y1_label]
    ax2.yaxis.set_minor_locator(FixedLocator(10/np.concatenate((np.arange(1,10,1), np.arange(0.1,1,0.1), np.arange(0.01,0.1,0.01)))))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    
    # 设置边框粗细
    bwith = 0.5
    ax2.spines['top'].set_linewidth(bwith)  # 设置上‘脊梁’为红色
    ax2.spines['right'].set_linewidth(bwith) # 设置上‘脊梁’为无色
    ax2.spines['bottom'].set_linewidth(bwith)
    ax2.spines['left'].set_linewidth(bwith)
    ax2.spines['top'].set_linewidth(bwith)
    ax2.spines['right'].set_linewidth(bwith)

    plt.tick_params(which='both', top=True,bottom=True,left=True,right=True, gridOn = True, grid_alpha = 0.0, width=0.5, size=1, grid_linewidth=0.5)
    plt.ylabel('Fidelity', fontdict={'family' : 'Arial', 'size'   : 8})
    plt.xlabel('sigma', fontdict={'family' : 'Arial', 'size'   : 8})

    plt.legend(labels=['Robust SLST','SLST'], loc='lower right', prop={'family' : 'Arial', 'size'   : 8}).set_zorder(100)
    plt.grid(True)
    plt.title('delta mean = '+str(sigma), fontdict={'family' : 'Arial', 'size'   : 8})
    plt.savefig("fid_simul_1vs_sigma_"+str(sigma)+".pdf", format="pdf")

    plt.show()
    return fig
    
if __name__ == '__main__':
    
    sigma = 0.1
    fid_RSLST = np.load('./result_data/fids_RSLST_vs_sigma_'+str(sigma)+'.npy')
    fid_SLST =  np.load('./result_data/fids_SLST_vs_sigma_'+str(sigma)+'.npy')
    lw = 0.5
    ms = 3
    cpthick = 0.5
    cpsize = 2


    plot(np.mean(fid_SLST, axis=2), np.mean(fid_RSLST, axis=2))
    
    
    
    
    
    