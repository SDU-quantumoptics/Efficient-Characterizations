# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, FixedLocator, NullFormatter

def plot_graph(x, fid):
    
    color5, color2, color3 =  tuple(np.array([50,130,191])/255), tuple(np.array([250,132,110])/255), tuple(np.array([200,36,35])/255)
    color4 = tuple(np.array([237, 85, 100])/255)
    
    color1, color2, color3 =  tuple(np.array([141, 160, 203])/255), tuple(np.array([102, 194, 165])/255),tuple(np.array([252, 141, 98])/255)
    
    colors = [color5, color2, color3, color4, color1]

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    
    fig = plt.figure(figsize=(1.8,1.4))
    ax = fig.add_subplot(111)
    
    for i in range(ob_num):
        plt.plot(x, fid[i, :],'o', markersize=ms,ls='-',linewidth=0.75, color=colors[i], zorder=3, label='k=, v=')
    
    
    plt.ylim(0.0,1.2)
    plt.xlim(0, x[-1])
    plt.tick_params(labelsize=7) 
    plt.yticks(ticks=[0, 0.5, 1], labels=['0', '0.5', '1.0'])
    if x[-1]>3.1:
        plt.xticks(ticks=[0, 1, 2, 3], labels=['0', '1', '2', '3'])
    else:
        plt.xticks(ticks=[0, 1, 2], labels=['0', '1', '2'])
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
    
    plt.ylabel('O', labelpad=1,fontdict={'family' : 'Arial', 'size'   : 8})
    plt.xlabel('Time (s)', labelpad=1, fontdict={'family' : 'Arial', 'size'   : 8})
    plt.grid(True)
    # plt.legend( ncol=1, loc='best', prop={'family' : 'Arial', 'size'   : 8},
    #         frameon=False).set_zorder(100)
    plt.show()
    # plt.savefig('./figure/real_time_ob5_new.pdf', format="pdf")
    return fig
    
if __name__ == '__main__':
    ms = 1.5
    ob_num = 5
    num = 0
    fidelity_ls = np.load('real_time_fidelity_5ob.npy')
    time_tagger = np.load('shadow_timetagger.npy')[:, num, :, 0]
    time_tagger_ms = time_tagger*10e-9
    state = 6 # choose the shadow of which state from 20 states
    
    x = time_tagger_ms[state, 0:-1:6]*1e-3
    y = fidelity_ls[:, 0:-1:6]
    plot_graph(x, y)


