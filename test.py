#-*- coding : utf-8 -*-
# coding: utf-8

import time
import numpy as np
import torch
import h5py
import functools
import math
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime
import parameters
from scipy import io
import sys
sys.path.append("..") 

print('current time:',datetime.now())

## GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

## import net
from source_transformation import wavenet2d as myNet
net = myNet()

def SNR(noisy,gt):
    res = noisy - gt
    msegt = np.mean(gt * gt)
    mseres = np.mean(res * res)
    SNR = 10 * math.log((msegt/mseres),10)
    return SNR

net.load_state_dict(torch.load(parameters.result_path+str(parameters.test_checkpoint_epoch)+'.pkl'))
net = net.to(device)

### test
time_span = parameters.timespan_input
time_span2 = parameters.timespan
receiver = parameters.receiver
trace = parameters.trace
testnum = parameters.testnum

## load testdata
f = h5py.File("./data/marmousi_testdata_shot77.h5", "r")
X = f['X'][:]
Y = f['Y'][:]
f.close()

Xinput = np.zeros([testnum,1,time_span,trace])

for k in range(1):
    for i in range(testnum):
        Xinput[k*testnum+i,0,:time_span2,:] = X[:,trace*i:trace*i+trace].reshape((1,1,time_span2,trace))
        
with torch.no_grad():
    Xt = Variable(torch.from_numpy(Xinput))
    Xt = Xt.to(device).type(torch.cuda.FloatTensor)
    Youtput = net(Xt).data.cpu().numpy()

Y_hat = np.zeros([time_span2,receiver])
for k in range(1):
    for i in range(testnum):
        Y_hat[:,i*trace:i*trace+trace] = Youtput[i][0]

## plot
import matplotlib.pyplot as plt
extent = [0, 1, 1, 0]
plt.figure(figsize=(14,5))
temp = parameters.sample_id_test
cmax = np.max(dat)
cmin = -cmax
colour = 'gray'

plt.subplot(1,3,1)     
plt.title('X, SNR=%.4f'%(SNR(X[temp],Y[temp])),fontsize=18)   
plt.imshow(X[temp],vmax=cmax,vmin=cmin,extent=extent,cmap=colour)
plt.yticks(size=15)
plt.xticks(size=15)

plt.subplot(1,3,2)     
plt.title('y_hat, SNR=%.4f'%(SNR(Y_hat[temp],Y[temp])),fontsize=18)
plt.imshow(Y_hat[temp],vmax=cmax,vmin=cmin,extent=extent,cmap=colour)
plt.yticks(size=15)
plt.xticks(size=15)

plt.subplot(1,3,3)     
plt.title('Y',fontsize=18)
plt.imshow(Y[temp],vmax=cmax,vmin=cmin,extent=extent,cmap=colour)
plt.yticks(size=15)
plt.xticks(size=15)

plt.tight_layout()
plt.savefig('')
plt.show()
