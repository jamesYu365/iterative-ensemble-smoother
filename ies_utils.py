# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 10:36:41 2022

@author: asus
"""

import numpy as np
from scipy import sparse
import pandas as pd
import networkx as nx
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import random
import os
import gc
from tqdm import tqdm
import shutil
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
#设置汉字格式
# sans-serif就是无衬线字体，是一种通用字体族。
# 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica,SimHei 中文的幼圆、隶书等等
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams.update({"font.size":16})#此处必须添加此句代码方可改变标题字体大小
plt.rcParams['xtick.direction'] = 'in'  # in; out; inout
plt.rcParams['ytick.direction'] = 'in'

import cv2
import aspose.words as aw
import cairosvg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM


#%%

def get_l1loss(output,target):
    loss=abs(output-target).mean()
    return loss


def configure_ies(ies_args,test_input,test_target):
    
    ########################################################
    obser=test_target
    measurement=obser.flatten()#B*Fout
    input_channel=test_input.shape[-1]
    output_channel=obser.shape[-1]
    ensemble=np.random.randn(ies_args['num_ensemble'],input_channel*test_input.shape[0])#E_B*Fin
    principal_sqrtR=np.diag(np.ones(output_channel))
    
    return obser,measurement,ensemble,principal_sqrtR,input_channel,output_channel


def get_output_loss(ensemble,measurement,input_channel,model,wbase,ne):
    sim_data=[]
    for i in range(ensemble.shape[0]):
        ensemble_tensor=torch.FloatTensor(ensemble[i:i+1,:]).reshape(-1,input_channel)
        sim_data.append(model(ensemble_tensor).detach().numpy().reshape(1,-1))
    sim_data=np.vstack(sim_data)
    sim_data=sim_data/np.tile(wbase,(sim_data.shape[0],1))
    
    obj=get_l1loss(sim_data[:ne,:],measurement)
    print('optimize objection=',obj)
    
    return sim_data,obj


def inner_iteration(ies_args,nd,ne,nf,input_channel,ensemble,measurement,sim_data,perturbed_data,wbase,model,
                    ud,wd,vd,svdpd,deltaM,deltaD,obj,lambd,iterat):
    
    iter_lambd=1#inner interation是对lambda进行迭代,然后使用iter_lambd控制内部迭代次数
    is_min_rn=0
    max_inn_iter=ies_args['max_in_iter']
    lambd_reduct=ies_args['lambd_reduct']
    lambd_incre=ies_args['lambd_incre']
    do_tsvd=ies_args['do_tsvd']
    min_rn=ies_args['min_rn']
    
    while iter_lambd<max_inn_iter:
        
        print('------inner interation step:',iter_lambd,'------')
        
        ensemble_old=ensemble.copy()
        sim_data_old=sim_data.copy()
        
        if do_tsvd:
            alpha=lambd*np.sum(wd**2)/svdpd
            x1=vd@sparse.diags(wd/(wd**2+alpha),0,(svdpd,svdpd))
            kgain=deltaM.T@x1@ud.T
            
        else:
            alpha=lambd*sum(sum(deltaD**2))/nd
            kgain=deltaM@deltaD/(deltaD@deltaD.t()+alpha*np.eye(nd))
        
        iterated_ensemble=ensemble[:ne,:]-(sim_data[:ne,:]-perturbed_data)@kgain.T
        ensemble_mean=iterated_ensemble.mean(axis=0)
        ensemble=np.vstack([iterated_ensemble,ensemble_mean])
        
        m_change=np.sqrt(np.sum((ensemble[:ne,:]-ensemble_old[:ne,:])**2)/nf)
        print('average change (in RMSE) of ensemble mean=',m_change)
        
        sim_data,obj_new=get_output_loss(ensemble,measurement,input_channel,model,wbase,ne)
        
        if obj_new>obj:
            lambd=lambd*lambd_incre
            print('lambd increase to',lambd)
            iter_lambd=iter_lambd+1
            sim_data=sim_data_old
            ensemble=ensemble_old
            
        else:
            lambd=lambd*lambd_reduct
            print('lambd reduce to',lambd)
            
            iterat=iterat+1
            
            if abs(obj_new-obj)/abs(obj)*100<min_rn:
                is_min_rn=1
                
            sim_data_old=sim_data
            ensemble_old=ensemble
            obj=obj_new
            break
    return iter_lambd,lambd,iterat,is_min_rn,ensemble,sim_data,obj
    
    
def outter_iteration(ies_args,nd,ne,nf,input_channel,init_obj,ensemble,measurement,sim_data,perturbed_data,wbase,model):
    iterat=0
    obj=init_obj
    init_lambd=ies_args['init_lambd']
    lambd=ies_args['init_lambd']
    beta=ies_args['beta']
    obj_thresh=beta**2*nd
    max_out_iter=ies_args['max_out_iter']
    max_inn_iter=ies_args['max_in_iter']
    lambd_incre=ies_args['lambd_incre']
    min_rn=ies_args['min_rn']
    max_lambd=ies_args['max_lambd']
    
    do_tsvd=ies_args['do_tsvd']
    tsvd_cut=ies_args['tsvd_cut']
    # flags of iES termination status; 1st => maxOuterIter; 2nd => objThreshold; 3rd => min_RN_change; 4th => max_lambd
    exit_flag=[0,0,0,0]
    objs=[]
    lambds=[]
    
    ########################################################
    while iterat<max_out_iter and obj>obj_thresh:
        
        print('------outer iteration step:',iterat,'------')
        print('number of measurement elements is ',measurement.size)
        
        #这里的deltaD和deltaM和matlab里面是互为转置关系
        deltaM=ensemble[:ne,:]-np.ones((ne,1))@ensemble[ne:,:]
        deltaD=sim_data[:ne,:]-np.ones((ne,1))@sim_data[ne:,:]
        
        if do_tsvd:
            ud,wd,vd=np.linalg.svd(deltaD.T,full_matrices=False)
            vd=vd.T
            wd=np.diag(wd)
            val=np.diag(wd)
            total=np.sum(val)
            for j in range(1,ne):
                svdpd=j
                if val[:j].sum()/total>tsvd_cut:
                    break
            
            print('svdpd=',svdpd)
            
            ud=ud[:,:svdpd]
            wd=val[:svdpd]
            vd=vd[:,:svdpd]
            
        iter_lambd,lambd,iterat,is_min_rn,ensemble,sim_data,obj=inner_iteration(ies_args,nd,ne,nf,input_channel,ensemble,measurement,sim_data,perturbed_data,wbase,model,
                                                                                ud,wd,vd,svdpd,deltaM,deltaD,obj,lambd,iterat)
        objs.append(obj)
        lambds.append(lambd)
        
        if iter_lambd>=max_inn_iter:
            
            lambd=lambd*lambd_incre
            if lambd<init_lambd:
                lambd=init_lambd
                
            iterat=iterat+1
            print('terminating inner iterations: iterLambda >= maxInnerIter')
            
            
        if is_min_rn:
            print('terminating outer iterations: reduction of objective function is less than ',min_rn)
            exit_flag[2]=1
            break
        
        if lambd>max_lambd:
            print('terminating outer iterations: lambd is bigger than ',max_lambd)
            exit_flag[3]=1
            break
        
    if iterat>=max_out_iter:
        print('terminating outer iterations: iter >= maxOuterIter')
        exit_flag[0]=1
        
    if obj<=obj_thresh:
        print('terminating outer iterations: obj <= objThreshold')
        exit_flag[1]=1
        
    print('exit_flag=',exit_flag)
    
    return ensemble,objs,lambds
        
        
#%%

def ies_main(ies_args,test_input,test_target,model):
    
    obser,measurement,ensemble,principal_sqrtR,\
    input_channel,output_channel=configure_ies(ies_args,test_input,test_target)
    
    ########################################################
    nd=len(measurement)#B*Fout,观测值的数量
    ne=ensemble.shape[0]#E,集成的个数
    
    ensemble_mean=ensemble.mean(axis=0)[np.newaxis,:]
    ensemble=np.vstack([ensemble,ensemble_mean])
    
    #B*Fout
    wbase=[]
    for i in range(obser.shape[0]):
        wbase.append(np.diag(principal_sqrtR))
    
    wbase=np.array(wbase).flatten()
    measurement=measurement/wbase
    
    perturbed_data=np.zeros((ne,nd))
    weight=ies_args['noise']*measurement
    # weight=np.ones_like(measurement)
    for i in range(ne):
        # perturbed_data[i,:]=measurement+weight*np.random.randn(*measurement.shape)
        perturbed_data[i,:]=measurement+weight*np.random.uniform(-1,1,measurement.shape)
        
    
    ########################################################
    nf=ensemble.shape[1]
    sim_data,obj=get_output_loss(ensemble, measurement, input_channel, model, wbase, ne)
    init_obj=obj
    ensemble,objs,lambds=outter_iteration(ies_args,nd,ne,nf,input_channel,init_obj,ensemble,measurement,sim_data,perturbed_data,wbase,model)
    
    ensemble=ensemble.reshape(ne+1,-1,input_channel)
    test_output=model(torch.FloatTensor(test_input)).detach().numpy()
    ensemble_output=model(torch.FloatTensor(ensemble[-1])).detach().numpy()
    objs=np.array(objs)
    lambds=np.array(lambds)
    return ensemble,test_output,ensemble_output,objs,lambds


