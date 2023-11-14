# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, FixedLocator, NullFormatter

def plot_graphs(error):
    
    
    color4 = tuple(np.array([237, 85, 100])/255)
    x = np.linspace(15, 315, 21, dtype='int32')
    

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    
    ################################################################################################################
    fig = plt.figure(figsize=(1.8,1.4))
    ax = fig.add_subplot(111)
    error_mean = np.mean(np.max(error_all[:, :, :], axis=2), axis=0)
    error_std = np.std(np.max(error_all[:, :, :], axis=2), axis=0)
    plt.plot(x, error_mean,'o', markersize=ms,ls='-',linewidth=0.5, color=color4, label='Octahedron (Exp.)')
    plt.errorbar(x, error_mean, yerr=error_std, ecolor=color4, linewidth=0
                  ,elinewidth=0.5, capsize=1.2, capthick=0.5, barsabove=False,color='black')
    
        
    plt.ylim(0.0,0.5)
    plt.xlim(0, 322)
    x_tick = x[np.arange(0,21,4)]
    plt.xticks(ticks=x_tick, labels=x_tick.astype('str'))
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
    
    plt.ylabel('MaxO|error|', labelpad=1,fontdict={'family' : 'Arial', 'size'   : 7})
    plt.xlabel('M', labelpad=1, fontdict={'family' : 'Arial', 'size'   : 8})
    plt.grid(True)
    plt.legend( ncol=1, loc='best', prop={'family' : 'Arial', 'size'   : 7},
                frameon=False).set_zorder(100)
    plt.show()
    # plt.savefig('max_error_1state.pdf', format="pdf")
    return fig
    
if __name__ == '__main__':
    ms = 2
    error_all = np.load('max_error_1state.npy')[:, :, 20::21]
    
    plot_graphs(error_all)
    

    