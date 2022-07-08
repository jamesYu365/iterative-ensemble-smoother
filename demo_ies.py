# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 19:31:46 2022

@author: asus
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from scipy import sparse
from torch.autograd import Variable
import time
import pandas as pd

import networkx as nx
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

import copy
import random
import pickle
import os

from ies_utils import ies_main

seed=1000
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(seed)


#%%

class linear_model(nn.Module):
    def __init__(self,input_channels,hidden_channels,output_channels):
        super(linear_model,self).__init__()
        
        self.layers=nn.Sequential(nn.Linear(input_channels,hidden_channels),
                                 nn.Dropout(p=0.1),
                                 nn.ReLU(),
                                 nn.Linear(hidden_channels,output_channels))
        
        

    def forward(self,x):
        x=self.layers(x)
        return x


model=linear_model(3,8,1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)
inputs=np.random.randn(100,3)
target=inputs[:,0:1]**3+2*inputs[:,1:2]**2-7*inputs[:,2:]**1
inputs=inputs+np.random.uniform(-1,1,inputs.shape)*inputs*0.2
num_epoch=1000
inputs=torch.FloatTensor(inputs)
target=torch.FloatTensor(target)
for i in range(num_epoch):
    output=model(inputs)
    loss=F.mse_loss(output,target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'epoch={i},loss={loss.item()}')
    
output=output.detach().numpy()
inputs=inputs.detach().numpy()
target=target.detach().numpy()
#%%

plt.plot(output,target,'ob')
plt.plot([output.min(),output.max()],[output.min(),output.max()])

#%%
#B_F
test_input=np.random.randn(60,3)
test_target=test_input[:,0:1]**3+2*test_input[:,1:2]**2-7*test_input[:,2:]**1

#%%

ies_args={}
ies_args['num_ensemble']=1000
ies_args['init_lambd']=1
ies_args['beta']=0.08
ies_args['max_out_iter']=50
ies_args['max_in_iter']=20
ies_args['lambd_reduct']=0.9
ies_args['lambd_incre']=1.2
ies_args['do_tsvd']=1
ies_args['tsvd_cut']=0.99
ies_args['min_rn']=0.01
ies_args['noise']=0.1
ies_args['max_lambd']=1e2

ensemble,test_output,ensemble_output,objs,lambds=ies_main(ies_args,test_input,test_target,model)


#%%

fig,axs = plt.subplots(2,2,figsize=(10,10),dpi=200)
axs[0,0].plot(ensemble[-1,:,:].flatten(),test_input.flatten(),'o')
axs[0,1].plot(ensemble_output.flatten(),test_output.flatten(),'o')
axs[0,1].plot([ensemble_output.min(),ensemble_output.max()],[ensemble_output.min(),ensemble_output.max()])

axs[1,0].plot(objs,label='objective')
axs[1,1].plot(lambds,label='lambda')
