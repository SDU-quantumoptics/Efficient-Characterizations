# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter


def plot_func(var_max_exp, var_max_sic):
    
    
    color5 =  tuple(np.array([50,130,191])/255)
    color4 = tuple(np.array([237, 85, 100])/255)
    color1, color2, color3 =  tuple(np.array([141, 160, 203])/255), tuple(np.array([102, 194, 165])/255),tuple(np.array([252, 141, 98])/255)
    fig = plt.figure(figsize=(1.8,1.4))
    ax = fig.add_subplot(111)
    

    x = np.arange(1, 2**7+1, 1)
    plt.plot(x, var_max_exp, marker='o',markersize=1.1,ls='None', linewidth=1, color=color4, zorder=3,label='Octahedron (Exp.)')
    plt.axhline(y = 1.5/2, color =color5, linestyle ="--",linewidth=0.75)
    plt.xticks(ticks=[ 10,  30,  50,  70,  90, 110], )
    plt.plot(x, var_max_sic, marker='D',markersize=1.0,ls='None', linewidth=1, color=color1, zorder=3,label='SIC (Sim.)')
    plt.ylim(0.4, 1.2)
    plt.xlabel('Observables', labelpad=1, fontdict={'family' : 'Arial', 'size' : 8})
    plt.ylabel('Maximum variance', labelpad=1, fontdict={'family' : 'Arial', 'size' : 8})
    plt.legend( ncol=1, loc='best', prop={'family' : 'Arial', 'size'   : 8},
            frameon=False).set_zorder(100)
        

        
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.tick_params(labelsize=7) 
    x1_label = ax.get_xticklabels() 
    [x1_label_temp.set_fontname('Arial') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels() 
    [y1_label_temp.set_fontname('Arial') for y1_label_temp in y1_label]
    ax.yaxis.set_minor_formatter(NullFormatter())
    bwith = 0.5
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    plt.tick_params(which='both', top=True,bottom=True,left=True,right=True, gridOn = True, grid_alpha = 0.0, width=0.5, size=1, grid_linewidth=0.5)
    plt.show()
    # plt.savefig("variance_list_all.pdf", format="pdf")
    

    
    
if __name__ == '__main__':
    variance_exp_ls = np.load('variance_exp_ls_315pho.npy')
    variance_sic_ls = np.load('variance_sic_ls_315pho.npy')

    var_mean_exp = np.mean(variance_exp_ls, axis=2)
    var_max_exp = np.max(var_mean_exp, axis=1)
    
    var_mean_sic = np.mean(variance_sic_ls, axis=2)
    var_max_sic = np.max(var_mean_sic, axis=1)
    

    plot_func(var_max_exp, var_max_sic) 

    






