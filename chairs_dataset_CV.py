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



class Dataset(torch.utils.data.Dataset):
    def __init__(self, train = True, n_pairs = None, sigma = 90., cross_validate = None, n_fold = 5, validate_set = False,
                 noise = None):
        
        np.random.seed(0)
        
        
        if train:
            self.data_dir = "/pghbio/dbmi/batmanlab/juc91/data/resized_chairs/train"
            self.info_file = "/pghbio/dbmi/batmanlab/juc91/data/resized_chairs/train_inf.txt"
        else:
            self.data_dir = "/pghbio/dbmi/batmanlab/juc91/data/resized_chairs/test"
            self.info_file = "/pghbio/dbmi/batmanlab/juc91/data/resized_chairs/test_inf.txt"
            
        
        data = np.loadtxt(self.info_file)
        
        self.label = data[:, 1]
        self.t = data[:, 3]
        
        #idx = np.where(self.t <= 180)[0]
        
        #self.file_list = [self.file_list[ii] for ii in idx]
        #self.t = self.t[idx]
        
        
        
        self.n_data = data.shape[0]

        if n_pairs is None:
            self.n_pairs = self.n_data * 5
        else:
            self.n_pairs = int (n_pairs * self.n_data + 1)

            
        self.N = max(self.n_data, self.n_pairs)
            
        self.idx_0 = np.random.choice(range(self.n_data), self.n_pairs)
        self.idx_1 = np.random.choice(range(self.n_data), self.n_pairs)
        
        t_1 = self.t[self.idx_0]
        t_2 = self.t[self.idx_1]
        
        dis = np.min( [ (t_1 - t_2) ** 2, (t_1 +360 - t_2) ** 2, (t_1 - 360 - t_2) ** 2 ], 0)
        
        self.S = np.exp(- dis / sigma ** 2)
        

        if not noise is None:
            self.S += np.random.randn(self.n_pairs) * noise
            self.S[self.S < 0] = 0
            self.S[self.S > 1] = 1
        
        
        self.transform = transforms.ToTensor()
        self.y = self.t

        
        self.validate_set = validate_set
        self.cross_validate = cross_validate
        
        if not cross_validate is None:
            self.n_fold = n_fold
            self.training_fold = list(range(n_fold))
            self.training_fold.remove(cross_validate)
            
            
            self.data_idx = np.arange(self.n_data)
            np.random.shuffle(self.data_idx)
            
            self.pair_idx = np.arange(self.n_pairs)
            np.random.shuffle(self.pair_idx)
            
            self.train_data_idx = []
            self.train_pair_idx = []
            
            for iii in self.training_fold:
                self.train_data_idx += list ( 
                            self.data_idx[ int( iii / self.n_fold * self.n_data) : 
                                      int ( ( iii + 1 ) / self.n_fold * self.n_data ) ] )
                self.train_pair_idx += list ( 
                           self.pair_idx[ int (iii / self.n_fold * self.n_pairs) : 
                                       int ( ( iii + 1 ) / self.n_fold * self.n_pairs ) ] )
                
            
            self.validate_data_idx = list ( 
                                self.data_idx[ int ( cross_validate / self.n_fold * self.n_data ) : 
                                       int ( ( cross_validate + 1 ) / self.n_fold * self.n_data ) ] 
                                          )
            
            self.validate_pair_idx = list ( 
                                self.pair_idx[ int ( cross_validate / self.n_fold * self.n_pairs ) : 
                                       int ( ( cross_validate + 1 ) / self.n_fold * self.n_pairs ) ]
                                          )
            
            
            if not self.validate_set:
                self.n_data = len( self.train_data_idx )
                self.n_pairs = len( self.train_pair_idx )
            else:
                self.n_data = len( self.validate_data_idx )
                self.n_pairs = len( self.validate_pair_idx )
        
            self.N = max(self.n_data, self.n_pairs)                
        
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, index):
        if self.cross_validate is None:
            idx = index % self.n_data
            idx0 = self.idx_0[index % self.n_pairs].item() 
            idx1 = self.idx_1[index % self.n_pairs].item() 
            S_idx = index % self.n_pairs

        else:
            if self.validate_set:
                idx = self.validate_data_idx [index % self.n_data]
                idx0 = self.idx_0[ self.validate_pair_idx [index % self.n_pairs] ]
                idx1 = self.idx_1[ self.validate_pair_idx [index % self.n_pairs] ]
                S_idx = self.validate_pair_idx [index % self.n_pairs]
            else:
                idx = self.train_data_idx [index % self.n_data]
                idx0 = self.idx_0[ self.train_pair_idx [index % self.n_pairs] ]
                idx1 = self.idx_1[ self.train_pair_idx [index % self.n_pairs] ]
                S_idx = self.train_pair_idx [index % self.n_pairs]        
                
                
        
        filename = join(self.data_dir, "{:06}.png".format(idx))
        filename_i = join(self.data_dir, "{:06}.png".format(idx0))
        filename_j = join(self.data_dir, "{:06}.png".format(idx1))

        return self.transform( Image.open( filename ) ),\
               self.transform( Image.open( filename_i ) ),\
               self.transform( Image.open( filename_j ) ),\
               self.S[S_idx]
    