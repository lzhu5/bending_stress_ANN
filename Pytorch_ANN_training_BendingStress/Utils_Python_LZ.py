# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:55:05 2021

@author: lzhu
"""

import matplotlib.pyplot as plt
import torch
import numpy as np

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
import sklearn.linear_model as lm

def Plotting(Y_ref, Y_pred, writer, nametag):
    fig = plt.figure()
    p1 = plt.plot(Y_ref)
    p2 = plt.plot(Y_pred)
    plt.title(nametag)
    plt.legend((p1[0], p2[0]), ('true', 'pred'))
    plt.show()
#    writer.add_figure(nametag, fig)

    # Hs_lim = 1.2
    fig = plt.figure()
#    fig.set_size_inches(10,10)
    plt.scatter(Y_ref,Y_pred)
    # plt.plot([0, Hs_lim],[0, Hs_lim],'k')
    plt.title(nametag)
    plt.axis('square')
#    ax[0,0].set_xlim(0,Hs_lim)
#    ax[0,0].set_ylim(0,Hs_lim)
    # m = np.linspace(0,Hs_lim,100)
    # reg = lm.LinearRegression().fit(Y_ref, Y_pred)
    # a1 = reg.coef_
    # b1 = reg.intercept_
    # n = (a1*m+b1).T
    # plt.plot(m,n,'r')
    # plt.show()
#    writer.add_figure(nametag, fig)


def RelativeError(model1, x_train_torch, y_train, lb_y, ub_y):
    with torch.no_grad():
        Hs_train_pred_torch = model1(x_train_torch)
        Hs_train_pred_norm = (Hs_train_pred_torch).detach().numpy() # convert from pytorch tensor to numpy array
        Hs_train_pred = (Hs_train_pred_norm + 1)/2*(ub_y[0] - lb_y[0]) + lb_y[0] # reverse-normalization
        err_Hs_train = np.linalg.norm(Hs_train_pred - y_train[:, 0:1])/np.linalg.norm(y_train[:, 0:1])
        rms_Hs_train = sqrt(mean_squared_error(Hs_train_pred, y_train[:, 0:1]))
        r2_Hs_train = (np.corrcoef(Hs_train_pred, y_train[:, 0:1],rowvar=False)[0, 1])**2
        SI_Hs_train = rms_Hs_train/np.nanmean(y_train[:, 0:1]);
        Bias_Hs_train = np.sum(Hs_train_pred - y_train[:, 0:1])/y_train[:, 0:1].size


    return err_Hs_train, Hs_train_pred, rms_Hs_train, r2_Hs_train, SI_Hs_train,Bias_Hs_train


class SineNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SineNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)
#        self.linear4 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h1 = torch.sin(self.linear1(x))
        h2 = torch.sin(self.linear2(h1))
        h3 = torch.sin(self.linear3(h2))
#        h4 = torch.sin(self.linear4(h3))
        return h3

class CustomizableNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out, No_HiddLayers):
        super(CustomizableNet, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(D_in, H)] + [torch.nn.Linear(H, H) for i in range(No_HiddLayers)] + [torch.nn.Linear(H, D_out)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i in range(len(self.linears) - 1): # except the output layer
            x = torch.tanh(self.linears[i](x))
        return self.linears[-1](x) # output layer
