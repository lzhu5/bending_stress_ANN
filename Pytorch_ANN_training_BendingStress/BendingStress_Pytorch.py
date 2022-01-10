# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 10:48:45 2021

@author: lzhu
"""

import sklearn.model_selection as sk
import torch
import scipy.io
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Utils_Python_LZ import *
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# fix random seed
torch.manual_seed(123)    # reproducible

# =============================================================================
# Prepare data
# =============================================================================
## load data
data = scipy.io.loadmat('./database.mat')
feature_data = data['input_database']
label_data = data['output_database']

x_train, x_vali, y_train_tmp, y_vali_tmp = \
sk.train_test_split(feature_data,label_data, test_size=0.2, random_state = 42)

y_train = np.log10(y_train_tmp)
y_vali = np.log10(y_vali_tmp)
# y_train = (y_train_tmp)
# y_vali = (y_vali_tmp)

# ## normalize input and output data by mapping each input/output feature to [-1, 1]
# # the normalization metric is chosen based on train&vali data
#
# compute normalization coeffcients
x = np.concatenate((x_train, x_vali), 0)

lb_x = x.min(axis = 0) # the minimum value of each input feature
ub_x = x.max(axis = 0) # the maximum value of each input feature

y = np.concatenate((y_train, y_vali), 0)
lb_y = y.min(axis = 0) # the minimum value of each output feature
ub_y = y.max(axis = 0) # the maximum value of each output feature

# implement normalization
x_train_norm = (x_train - lb_x)/(ub_x - lb_x)*2.0 - 1.0
y_train_norm = (y_train - lb_y)/(ub_y - lb_y)*2.0 - 1.0

x_vali_norm = (x_vali - lb_x)/(ub_x - lb_x)*2.0 - 1.0
y_vali_norm = (y_vali - lb_y)/(ub_y - lb_y)*2.0 - 1.0

# x_test_norm = (x_test - lb_x)/(ub_x - lb_x)*2 - 1
#
## convert numpy array to pytorch tensors
x_train_torch = torch.from_numpy(x_train_norm).float()
y_train_torch = torch.from_numpy(y_train_norm).float()

# print(x_train_norm)
# print(x_train_torch)

x_vali_torch = torch.from_numpy(x_vali_norm).float()
y_vali_torch = torch.from_numpy(y_vali_norm).float()

# x_test_torch = torch.from_numpy(x_test_norm).float()

# =============================================================================
# Design fully connected NN
# =============================================================================
## input to model 1: u, v, water depth
D_in = x_train.shape[1] # get number of features
H = 50
D_out = 1

# model 1:tanh net
model1 = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H,H),
    torch.nn.Tanh(),
    torch.nn.Linear(H,H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
#    torch.nn.Softplus()
)
# initialization for tanh net
#torch.nn.init.xavier_normal_(model1[0].weight)
#torch.nn.init.zeros_(model1[0].bias)
#torch.nn.init.xavier_normal_(model1[2].weight)
#torch.nn.init.zeros_(model1[2].bias)
#torch.nn.init.xavier_normal_(model1[4].weight)
#torch.nn.init.zeros_(model1[4].bias)

# =============================================================================
# Train models
# =============================================================================
# define training logger
# Writer will output to ./runs/ directory by default
#pip install tensorboard

#in Anaconda prompt,
#conda activate pytorch
#cd Desktop/BRT_python
#tensorboard --logdir=runs
writer = SummaryWriter()

# print(list(model1.parameters()))

# optimizer = torch.optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(list(model1.parameters()), lr=1e-3, weight_decay=0, )
#optimizer = torch.optim.LBFGS(model1.parameters(), history_size=10, max_iter=4)
# optimizer = torch.optim.RMSprop(model1.parameters(), lr=1e-3)

loss_fn = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.05)

print("Begin training.")
for epoch in tqdm(range(5000)):

    # train loss for model 1
    bendingstress_pred_torch = model1(x_train_torch)
    loss1 = loss_fn(bendingstress_pred_torch, y_train_torch) # train loss

    train_loss = loss1


    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    train_loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        if epoch % 10 == 0:
            # vali loss for model 1
            bendingstress_vali_torch = model1(x_vali_torch)
            loss1_vali = loss_fn(bendingstress_vali_torch, y_vali_torch) # vali loss

            # record train loss and vali loss
            writer.add_scalars('Loss1', {'train': loss1,
                                         'vali': loss1_vali}, epoch)

# # =============================================================================
# # result evaluation
# # =============================================================================
# ## compute relative error
# # train error
err_Hs_train, Hs_train_pred, rms_Hs_train, r2_Hs_train, SI_Hs_train, Bias_Hs_train  = RelativeError(model1, x_train_torch, y_train, lb_y, ub_y)
# #writer.add_text('err_Hs_train', str(err_Hs_train))
# #writer.add_text('err_Tp_train', str(err_Tp_train))
#
# #print('rms_Hs_train: ' + str(rms_Hs_train))
# #print('rms_Tp_train: ' + str(rms_Tp_train))
# #print('r2_Hs_train: ' + str(r2_Hs_train))
# #print('r2_Tp_train: ' + str(r2_Tp_train))
#
# # vali error
err_Hs_vali, Hs_vali_pred, rms_Hs_vali, r2_Hs_vali, SI_Hs_vali, Bias_Hs_vali  = RelativeError(model1, x_vali_torch, y_vali, lb_y, ub_y)
# #writer.add_text('err_Hs_vali', str(err_Hs_vali))
# #writer.add_text('err_Tp_vali', str(err_Tp_vali))
#
# #print('rms_Hs_vali: ' + str(rms_Hs_vali))
# #print('rms_Tp_vali: ' + str(rms_Tp_vali))
# #print('r2_Hs_vali: ' + str(r2_Hs_vali))
# #print('r2_Tp_vali: ' + str(r2_Tp_vali))
#
# # # test error
# # err_Hs_test, Hs_test_pred, rms_Hs_test, r2_Hs_test, SI_Hs_test, Bias_Hs_test  = RelativeError(model1, x_test_torch, y_test, lb_y, ub_y)
# # #writer.add_text('err_Hs_test', str(err_Hs_test))
# # #writer.add_text('err_Tp_test', str(err_Tp_test))
# # print('rms_Hs_test: ' + str(rms_Hs_test))
# # print('r2_Hs_test: ' + str(r2_Hs_test))
#
# # file1 = open("NNsPerformance.txt","a")
# #
# # file1.write(str(rms_Hs_test) + '\n')
# # file1.write(str(r2_Hs_test) + '\n')
# # file1.write(str(Bias_Hs_test) + '\n')
# # file1.write(str(SI_Hs_test) + '\n')
# # file1.close()
#
#
## visualize discrepancy
Plotting(y_train[:, 0:1], Hs_train_pred, writer, 'train')

Plotting(y_vali[:, 0:1], Hs_vali_pred, writer, 'vali')

# Plotting(y_test[:, 0:1], Hs_test_pred, writer, 'Hs_test')

## save prediction to matlab file
#scipy.io.savemat('Gandy_Python_output_WG3.mat', {'Hs_train_pred': Hs_train_pred, 'Tp_train_pred': Tp_train_pred,
#                                    'Hs_vali_pred': Hs_vali_pred, 'Tp_vali_pred': Tp_vali_pred,
#                                    'Hs_test_pred': Hs_test_pred, 'Tp_test_pred': Tp_test_pred,
#                                    'y_train': y_train,'y_vali': y_vali,'y_test': y_test})

writer.close()
