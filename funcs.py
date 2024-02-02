import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_data(data):
    cmax = 1
    cmin = -cmax
    colour = 'seismic'
    plt.figure(figsize=(8,9))

    x_major_locator=MultipleLocator(100)
    y_major_locator=MultipleLocator(0.5)
    ax=plt.gca()
    ax.set_xlabel('Distance (km)',fontsize=18)
    ax.set_ylabel('Time (s)',fontsize=18)

    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('left')
    ax.set_xticks([0,200,400,600,800,920])
    ax.set_xticklabels(['0.0','2.0','4.0','6.0','8.0','9.2'],rotation = 0,fontsize = 16)
    ax.set_yticks([0,1389,2778,4167,5556])
    ax.set_yticklabels(['0.0','1.0','2.0','3.0','4.0'],rotation = 0,fontsize = 16)

    plt.imshow(data,cmap=colour,vmax=cmax,vmin=cmin, aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.3)
    cb = plt.colorbar(cax=cax)
    cb.ax.tick_params(labelsize=16)

    plt.tight_layout()
    
def L2loss(noisy,gt):
    res = noisy - gt
    msegt = np.mean(gt * gt)
    mseres = np.mean(res * res)
    loss = mseres/msegt
    return loss