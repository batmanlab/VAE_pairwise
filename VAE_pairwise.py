#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:57:32 2019

@author: juc91
"""

import PIL
from PIL import Image
import os
from os.path import isfile, join 
from scipy.io import loadmat

import torch
import torch.utils.data
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from torch.distributions.categorical import Categorical

    
    
class VAE_pairwise(nn.Module):
    
    def __init__(self, Dataset, K = 2, n_channels = 128, n_conv = 4, n_fc = 1, d_latent = 128, 
                 alpha = 1., gamma = 1., C1 = 1., C2 = .1, beta1 = 1., beta2 = 1., 
                 sigma = 45., cross_validate = None, noise = None, n_pairs = None, 
                 batch_size = 32, lr = 1e-4, filename = "./model", 
                 annealing = True, annealing_step = 2000, training_step = 100000000, 
                 device="cuda"):
        super(VAE_pairwise, self).__init__()

        self.train_data = Dataset(True, sigma = sigma, cross_validate = cross_validate, 
                                  validate_set = False, noise = noise, n_pairs = n_pairs)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, 
                            shuffle = True, batch_size = batch_size, num_workers = 0)
        
        if not cross_validate is None:
            self.validate_data = Dataset(True, sigma = sigma, cross_validate = cross_validate, validate_set = True)
            self.validate_loader = torch.utils.data.DataLoader(self.validate_data, 
                            shuffle = True, batch_size = batch_size, num_workers = 0)
            
        
        self.test_data = Dataset(False, sigma = sigma)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, 
                            shuffle = False, batch_size = batch_size, num_workers = 0)
            
        img_channels, self.n_pix, _ = self.train_data[0][0].shape


        if len(torch.unique(next(iter(self.train_loader))[3])) > 2:
            self.binary_similarity = False
        else:
            self.binary_similarity = True
            
        self.d_latent = d_latent
        self.n_conv = n_conv
        self.n_fc = n_fc
        
        self.n_last_channels = n_channels * 2 ** (self.n_conv - 1)
        self.d_last_image = self.n_pix // 2 ** self.n_conv 
        self.d_fc = self.n_last_channels * self.d_last_image ** 2
        
        
        self.conv = nn.ModuleList()
        self.conv_bn = nn.ModuleList()
        
        input_channel = img_channels
        output_channel = n_channels
        
        for ii in range(n_conv):
            self.conv.append(nn.Conv2d(input_channel, output_channel, 4, 2, 1))
            self.conv_bn.append(nn.BatchNorm2d(output_channel))
            input_channel = 2 ** (ii) * n_channels
            output_channel = 2 ** (ii+1) * n_channels
        
        
            
        self.deconv = nn.ModuleList()
        self.deconv_bn = nn.ModuleList()
        
        for ii in range(n_conv):
            iii = n_conv - ii - 1
            input_channel = 2 ** (iii ) * n_channels
            output_channel = 2 ** (iii - 1) * n_channels
            
            if iii == 0:
                output_channel = img_channels

            self.deconv.append(nn.ConvTranspose2d(input_channel, output_channel, 4, 2, 1))
            self.deconv_bn.append(nn.BatchNorm2d(input_channel))
            

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for ii in range(n_fc):
            if ii == 0:
                self.encoders.append(nn.Linear(self.d_fc, self.d_latent))
            else:
                self.encoders.append(nn.Linear(self.d_latent, self.d_latent))
            
        self.mu_layer = nn.Linear(self.d_latent, self.d_latent)
        self.var_layer = nn.Linear(self.d_latent, self.d_latent)
        
        for ii in range(n_fc + 1):
            if ii < n_fc:
                self.decoders.append(nn.Linear(self.d_latent, self.d_latent))
            else:
                self.decoders.append(nn.Linear(self.d_latent, self.d_fc))
        
        self.K = K
        
        self.alpha = alpha
        
        self.beta1 = beta1
        self.beta2 = beta2

        self.gamma0 = gamma
        
       
        self.C1 = C1
        self.C2 = C2
           
        
        self.filename = filename
        self.annealing = annealing
        self.annealing_step = annealing_step
        self.training_step = training_step
        
        
        self.device = device
        
        
                                                        
        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 0.)
    
        self.training_loss = []
        self.test_loss = []
        self.n_step = 0
        
        torch.cuda.manual_seed(0)

    
    def encode(self, X):
        res = X
        for ii in range(self.n_conv):
            res = F.leaky_relu_(self.conv_bn[ii](self.conv[ii](res)))
        res = res.view(-1, self.d_fc)
        for M in self.encoders:
            res = F.leaky_relu_(M(res))
        return self.mu_layer(res), self.var_layer(res)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)
    
    def decode(self, Z):
        res = Z
        for M in self.decoders[:]:
            res = F.leaky_relu_(M(res))
        
        res = res.view(-1, self.n_last_channels, self.d_last_image, self.d_last_image)
        
        for ii in range(self.n_conv - 1):
            res = F.leaky_relu_(self.deconv[ii]( self.deconv_bn[ii](res) ))

        res = torch.sigmoid( self.deconv[-1]( self.deconv_bn[-1](res) ) )
        
        return res

    def forward(self, x):
        mu, log_var = self.encode(x)
        Z = self.reparameterize(mu, log_var)
        return mu, log_var, self.decode(Z), Z
    
    
    def recon_loss(self, recon_x, x):
        #criterion = nn.MSELoss(reduction = 'sum')
        criterion = nn.BCELoss(reduction = 'sum')
        return criterion(recon_x, x)
    
    def KLD(self, mu, log_var):
        
        tmp_KLD = .5 * (-1 - log_var + log_var.exp() + mu.pow(2)  )
        KLD_1 = tmp_KLD[:, :self.K].sum()
        KLD_2 = tmp_KLD[:, self.K:].sum()
        
        return self.beta1 * KLD_1 + self.beta2 * KLD_2
        #return .5 * (-1 - log_var + log_var.exp() + mu.pow(2)  ).sum()

    def VAE_loss_function(self, recon_x, x, mu, log_var):
        return self.recon_loss(recon_x, x) + self.KLD(mu, log_var)
    
    
    def Classifier_loss(self, X_i, X_j, y_ij):
        mu_i, log_var_i = self.encode(X_i)
        Z_i = self.reparameterize(mu_i, log_var_i)
        mu_j, log_var_j = self.encode(X_j)
        Z_j = self.reparameterize(mu_j, log_var_j)
        
        #t = self.C1 * ( torch.norm(Z_i[:, -self.K:] - Z_j[:, -self.K:], dim = 1) - torch.abs(self.C2) )
        diff = ( Z_i[:, :self.K] - Z_j[:, :self.K] ) ** 2
        
        
        
        # t = self.W (diff) - self.C2
        if self.annealing and self.n_step < self.annealing_step:
            t = self.C1 * ( diff.sum(1) - self.C2 )
        else:
            t = self.C1 * ( diff.sum(1) - self.C2 )

        if self.binary_similarity:
            return - ( y_ij * F.logsigmoid( -t ) + (1 - y_ij) * F.logsigmoid( t ) ).sum()
        else:
            log_g = F.logsigmoid( t ) 
            log_g_I = F.logsigmoid( -t )
            return - ( 
                         y_ij * log_g_I + (1 - y_ij) * log_g
                         - torch.log(
                             ( 1 - 2 * torch.sigmoid( t ) ) / 
                             (log_g_I - log_g)
                         )
                     ).sum() 
        



    def PD_norm(self, VAE_loss, Z):
        return (
            torch.autograd.grad(VAE_loss, Z,  create_graph=True)[0][:, :-self.K] ** 2
        ).sum()
        
        
    def total_loss(self, X, X_i, X_j, y_ij):
        
        N = X.shape[0]

        
        C_loss = self.Classifier_loss(X_i, X_j, y_ij)
        
        
        mu, log_var, X_recon, Z = self.forward(X)

        #PD_loss = self.PD_norm(VAE_loss, Z)
        
        loss = self.gamma * self.recon_loss(X_recon, X) + self.KLD(mu, log_var) + self.alpha * C_loss  #+ self.gamma * PD_loss
        
        return loss
    
    def train_model(self, epochs = 20, step = None):
        
        print ("epochs:", epochs)
        print ("training_step", self.training_step)
        
        self.train()
        
        if not self.annealing:
            self.gamma = self.gamma0
        
        for iii in range(epochs):
            

            ii = 0
            n_data = 0
            total_loss = 0.

            for data_batch in self.train_loader:
                
                if self.n_step > self.training_step:
                    break

                X = data_batch[0].to(self.device)
                X_i = data_batch[1].to(self.device)
                X_j = data_batch[2].to(self.device)
                y_ij = data_batch[3].float().to(self.device)
                
                if self.annealing:
                    rho = torch.sigmoid(
                            torch.tensor( (self.n_step - self.annealing_step) / (self.annealing_step / 10) ) 
                        ).to(self.device)
                    self.gamma =  self.gamma0 * rho
                
                
                batch_size = X.shape[0]
                
                loss = 0

                self.optimizer.zero_grad()
                
                loss += self.total_loss(X, X_i, X_j, y_ij)

                loss.backward()

                self.optimizer.step()
                
                
                total_loss += loss.item()
                n_data += batch_size


                
                if not step is None:
                    if ii > step:
                        break

                if ii % 100 == 0:
                    print ( iii, self.n_step, total_loss / n_data, "\t", loss.item() / batch_size
                          )
                    print ( "Classifier Loss",  self.Classifier_loss(X_i, X_j, y_ij).item(),
                            "gamma={}".format(self.gamma), 
                            "C1 = {}".format(self.C1), 
                            "C2 = {}".format(self.C2)
                          )
                    
                    torch.save(self.state_dict(), self.filename)
                ii +=1
                self.n_step += 1
                    

            if self.n_step >= self.training_step:
                print("Finished training.", self.n_step, self.training_step)
                torch.save(self.state_dict(), self.filename)
                break
                



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           