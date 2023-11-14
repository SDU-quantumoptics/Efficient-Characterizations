# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter


def plot_func(var_x_ls):
    
    photon_ls = np.linspace(15, 315, 21, dtype='int32')
    x = photon_ls
    var_x_max = np.max(var_x_ls, axis=1)
    var_x_max_mean = np.mean(var_x_max, axis=1) 
    var_x_max_std = np.std(var_x_max, axis=1)
    
    color5, color2, color3 =  tuple(np.array([50,130,191])/255), tuple(np.array([250,132,110])/255), tuple(np.array([200,36,35])/255)
    color4 = tuple(np.array([237, 85, 100])/255)
    
    fig = plt.figure(figsize=(1.8,1.4))
    ax = fig.add_subplot(111)
    


    plt.plot(x, var_x_max_mean, marker='o',markersize=2,ls='-', 
             linewidth=0.5, color=color4, zorder=3, label='Octahedron (Exp.)')
    plt.errorbar(x, var_x_max_mean, yerr=var_x_max_std, ecolor=color4, linewidth=0
                  ,elinewidth=0.5, capsize=1.2, capthick=0.5, barsabove=False,color='black', zorder=3)
    
    plt.axhline(y = 0.75, color =color5, linestyle ="--",linewidth=0.75)
    plt.ylim(0.4, 1.2)
    plt.xlabel('M', labelpad=1,fontdict={'family' : 'Arial', 'size' : 8})
    plt.ylabel('Maximum variance', labelpad=1, fontdict={'family' : 'Arial', 'size' : 8}) 
    x_tick = x[np.arange(0,21,4)]
    plt.xticks(ticks=x_tick, labels=x_tick.astype('str'))
    plt.legend( ncol=1, loc='lower center', prop={'family' : 'Arial', 'size'   : 8},
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
    # plt.savefig("variance_x_M_errorbar_new.pdf", format="pdf")
    # return fig
    

    
    
if __name__ == '__main__':
    variance_exp_ls = np.load('variance_exp_x_M_errorbar.npy')

    plot_func(variance_exp_ls) 

    






