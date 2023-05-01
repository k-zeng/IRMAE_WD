#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:29:26 2023

@author: kevin
"""

import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import re
import IRMAE_WD


import seaborn as sns
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# For outputting info when running on compute nodes
def output(text):
    # Output epoch and Loss
    name='Out_Test.txt'
    newfile=open(name,'a+')
    newfile.write(text)
    newfile.close()

#Need to get main pathing for autoencoder batch. This gets AE model
AE_path = './models/'
NODE_path = './models/'
sys.path.insert(0,AE_path)

###############################################################################
# Arguments from the submit file
###############################################################################
parser = argparse.ArgumentParser('ODE demo')
# These are the relevant sampling parameters
parser.add_argument('--data_size', type=int, default=100)  #IC from the simulation
parser.add_argument('--dt',type=float,default=0)
parser.add_argument('--sim_time', type=float, default=10)  #Time of simulation to draw samples from
parser.add_argument('--batch_time', type=int, default=9)   #Samples a batch covers (this is 10 snaps in a row in data_size)
parser.add_argument('--batch_size', type=int, default=20)   #Number of IC to calc gradient with each iteration

parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--niters', type=int, default=60)       #Iterations of training
parser.add_argument('--test_freq', type=int, default=20)    #Frequency for outputting test loss
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

#Add 1 to batch_time to include the IC
args.batch_time+=1 

from torchdiffeq import odeint_adjoint as odeint

# Check if there are gpus
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

###############################################################################
# Classes
###############################################################################

# This is the class that contains the NN that estimates the RHS
class ODEFunc(nn.Module):
    def __init__(self,trunc):
        super(ODEFunc, self).__init__()
        # Change the NN architecture here
        self.net = nn.Sequential(
            nn.Linear(trunc, 200),
            nn.Sigmoid(),
            nn.Linear(200, 200),
            nn.Sigmoid(),
            nn.Linear(200,200),
            nn.Sigmoid(),
            nn.Linear(200, trunc),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # This is the evolution with the NN
        return self.net(y)

# This class is used for updating the gradient
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

###############################################################################
# Functions
###############################################################################
# Gets a batch of y from the data evolved forward in time (default 20)
def get_batch(t,true_y):
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

def plotting(true_y, pred_y,xlabel='x',ylabel='y',zlabel='z'):
    plt.figure(figsize=(7.5,6))
    ax=plt.subplot(projection='3d')
    plt.plot(true_y.detach().numpy()[:, 0, 0],true_y.detach().numpy()[:, 0, 1],true_y.detach().numpy()[:, 0, 2],'.',color='black',linewidth=.5,markersize=1,alpha=1)
    plt.plot(pred_y.detach().numpy()[:, 0, 0],pred_y.detach().numpy()[:, 0, 1],pred_y.detach().numpy()[:, 0, 2],'.',color='red',linewidth=.5,markersize=1,alpha=.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend(('True','Pred'))

def best_path(path):

    text=open(path+'Trials.txt','r')
    MSE=[]
    for line in text:
        vals=line.split()
        # Check that the line begins with T, meaning it has trial info
        if vals[0][0]=='T':
            # Put MSE data together
            MSE.append(float(vals[1]))
    
    idx=np.argmin(np.asarray(MSE))+1

    return path+'Trial'+str(idx)

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def pdf_calc(h,bins_x,bins_y):

    count=np.zeros((len(bins_x),len(bins_y)))
    
    #adding the weight was messing things up because dx and dxx are a different size then dt
    for x,y in zip(np.nditer(h[:,0]),np.nditer(h[:,1])):
        idx_x=np.argmin((bins_x-x)**2)
        idx_y=np.argmin((bins_y-y)**2)
        count[idx_x,idx_y]+=1
        
    dem=0
    bin_area=(bins_x[1]-bins_x[0])*(bins_y[1]-bins_y[0])
    for i in range(len(bins_x)):
        for j in range(len(bins_y)):
            dem+=count[i,j]*bin_area            
    
    pdf=count/dem
    
    return pdf

if __name__=='__main__':
    ###########################################################################
    # Load Data
    ###########################################################################
    # Load the data and put it in the proper form for training
    u = scipy.io.loadmat('./dataset/L22.mat')
    u = u['ut']
    u = u[0:64,0:40000]
    u = u.T
    [M,N]=u.shape
    ts=np.arange(0.0, 100.0, 0.25)

    # Normalize data by subtracting off the mean and dividing by the standard deviation
    u_mean=np.mean(u,axis=0)
    u_std=np.std(u,axis=0)
    u=(u-u_mean[np.newaxis,:])/u_std[np.newaxis,:]

    ###########################################################################
    # Load Encoder and decoder
    ###########################################################################
    #Assumes that the main NODE directory is the same level main AE directory
    ae_model = IRMAE_WD.autoencoder().to(device)
    ae_model.double()

    ae_model.load_state_dict(torch.load(AE_path+'/NeuralODE/IRMAEWD_AE.pt'))
    
    #Number of Manifold SVs to truncate to: KSE L=22 is 8
    trunc = 8

    Testdata = torch.tensor(u, dtype=torch.double).to(device)
    z = ae_model.encode(Testdata)
    code_data = z.detach().numpy()

    #load basis vectors and project
    code_U, code_S, code_V = pickle.load(open(AE_path+'/NeuralODE/code_svd.p','rb'))
    print(code_U.shape)
    code_mean, code_std = pickle.load(open(AE_path+'/NeuralODE/code_musigma.p','rb'))
    U_trunc = code_U[:,:trunc]
    code_clean = (code_data - code_mean) #/code_std
    print(code_clean.shape)

    #project onto leading values
    h_std = pickle.load(open(NODE_path+'NeuralODE/h_std.p','rb'))
    h = code_clean @ U_trunc
    h = h / h_std
    h_full = (code_data - code_mean) @ code_U
    
    
    np.random.seed(66)
    ex=np.random.randint(int(M/2))
    h0 = torch.tensor(h[ex:ex+1,:], dtype=torch.double).to(device)
    #h0 = ae_model.encode(IC_snap)

    ###########################################################################
    # Evolve data forward with odenet
    ###########################################################################
    # Load odenet
    g=torch.load(NODE_path+'/NeuralODE/IRMAEWD_dynamics.pt') #NEED TO CHECK FOR FUTURE

    # Format IC
    t=torch.tensor(ts)    
    t=t.type(torch.FloatTensor)
    h0=torch.tensor(h0)
    h0=h0.type(torch.FloatTensor)

    #odenet (I have to run it in chunks because the normal odenet has memory leakage issues)!!!!
    hNN = odeint(g, h0, t)
    hNN=hNN.detach().numpy()[:, 0, :]
    hNN = hNN * h_std
    
    z_rec = hNN @ U_trunc.T 
    z_rec = z_rec + code_mean

    # Decode
    zNN = torch.tensor(z_rec, dtype=torch.double).to(device)
    uNN = ae_model.decode(zNN).detach().numpy() *u_std+u_mean
    zNN = zNN.detach().numpy()

    ###########################################################################
    # Plotting
    ###########################################################################
    #Small chunck of data just to plot (yes technically this isnt test data)
    n_snaps = len(ts)
    u_truth = u[ex:ex+n_snaps,:] #remember "u" has std and mean removed
    h = h * h_std
    h_truth = h[ex:ex+n_snaps,:]
    h_truth_full = h_full[ex:ex+n_snaps,:]

    h_pred = hNN[0:n_snaps,:]
    print(h_pred.shape)
    NODE_rec_data = uNN[0:n_snaps,:] #This has std and mean returned
    Full_data = u_truth*u_std+u_mean #need to return std and mean
    t_data = ts[0:n_snaps]

    ###########################################################################
    # Plot Latent Spaces
    ###########################################################################
    
    #Setting Plotting Variables
    colormap = sns.diverging_palette(240, 10, as_cmap=True)
    colormap_latent = sns.diverging_palette(220, 5, s=90,l=65, as_cmap=True)
    N = 64
    L = 22
    max_steps = n_snaps
    dt = 0.05
    AC = 5
    x_grid = np.arange(N)*L/N
    x_grid_ticks = np.arange(0,22,10)
    h_grid = np.arange(trunc+1) + 0.5
    h_grid_ticks = np.arange(0,trunc+1,2) + 0.5
    h_full_grid = np.arange(20+1) + 0.5
    h_full_grid_ticks = np.arange(0,20+1,2) + 0.5
    t_log = t_data
    th_log = t_data
    linec = 'black'
    countc = 4

    h_truth_plot = np.hstack((h_truth, h_truth[:,0,np.newaxis]))
    h_pred_plot = np.hstack((h_pred, h_pred[:,0,np.newaxis]))
    h_full_truth_plot = np.hstack((h_truth_full, h_truth_full[:,0,np.newaxis]))
    
    #Generate 4 Plots
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 16})
    ax1 = plt.subplot(221)
    a = plt.pcolormesh(t_log,x_grid,Full_data.T,cmap=colormap,vmin=-3, vmax=3,shading='gouraud')
    plt.contour(t_log,x_grid,Full_data.T, countc, colors=linec,linewidths=1,vmin=-3, vmax=3)
    plt.setp(ax1.get_xticklabels(), visible=False)
    a_bar=plt.colorbar(a)
    a_bar.set_label(r'$u_{truth}$',rotation=270, labelpad=10)
    plt.yticks(x_grid_ticks)
    ax1.set_ylabel(r'$x$')

    ax2 = plt.subplot(223)
    b = plt.pcolormesh(t_log,x_grid,NODE_rec_data.T,cmap=colormap,vmin=-3, vmax=3,shading='gouraud')
    plt.contour(t_log,x_grid,NODE_rec_data.T, countc, colors=linec,linewidths=1,vmin=-3, vmax=3)
    b_bar = plt.colorbar(b)
    b_bar.set_label(r'$D(h_{i,pred})$',rotation=270, labelpad=10)
    ax2.set_ylabel(r'$x$')
    ax2.set_xlabel(r'$t$')
    plt.yticks(x_grid_ticks)

    ax3 = plt.subplot(222)
    c = plt.pcolormesh(t_log, h_grid,h_truth_plot.T,cmap=colormap_latent,vmin=-1, vmax=1)
    plt.yticks(h_grid_ticks[:-1]+0.5)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_ylabel(r'$i$')
    c_bar = plt.colorbar(c)
    c_bar.set_label(r'$h_{i,truth}$',rotation=270, labelpad=10)
    

    ax4 = plt.subplot(224)
    d = plt.pcolormesh(t_log,h_grid,h_pred_plot.T,cmap=colormap_latent,vmin=-1, vmax=1,shading='nearest')
    plt.yticks(h_grid_ticks[:-1]+0.5)
    ax4.set_ylabel(r'$i$')
    ax4.set_xlabel(r'$t$')
    d_bar = plt.colorbar(d)
    d_bar.set_label(r'$h_{i,pred}$',rotation=270, labelpad=10)

    fig.set_size_inches(10,4)
    plt.tight_layout()
    fig.savefig('KSE_Forecasting.png', dpi=300)




            