#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:53:25 2019

@author: juc91
"""
import PIL
from PIL import Image
import os
from os.path import isfile, join 
from scipy.io import loadmat



import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision.transforms.functional
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from importlib import reload
from scipy.spatial.distance import pdist
import torch.utils.data
from torch.distributions.categorical import Categorical


from sklearn.linear_model import LogisticRegression

class Dataset(torch.utils.data.Dataset):
    def __init__(self, train = True, n_pairs = None, sigma = 60.,  cross_validate = None, n_fold = 5, validate_set = False, 
                 noise = None):
        
        np.random.seed(0)
        
        f = open( "/pghbio/dbmi/batmanlab/juc91/data/cars/list.txt" )
        self.file_list = f.readlines()
        f.close()

        data_dir = "/pghbio/dbmi/batmanlab/juc91/data/cars/"
        
        
        self.train_list = self.file_list[:146]
        self.test_list = self.file_list[146:]
        
        if train:
            self.file_list = self.train_list
        else:
            self.file_list = self.test_list
        
        self.n_data = len(self.file_list) * 24 * 4
        
        self.X = None
        
        M_ToImg = torchvision.transforms.ToPILImage()
        M_Totensor = torchvision.transforms.ToTensor()
        M_resize = torchvision.transforms.Resize( (64, 64) )

        
        for fff in self.file_list:
            
            
            data_loaded = torch.tensor( loadmat( 
                            join(data_dir, "{}.mat".format(fff[:-1]) ) 
                        )["im"]
                        ).view(128, 128, 3, -1).transpose(0, 3).transpose(1, 2).transpose(2, 3)
            
            data = torch.zeros(data_loaded.shape[0], 3, 64, 64)
            
            for iii in range(data_loaded.shape[0]):
                data[iii] = M_Totensor(M_resize(M_ToImg(data_loaded[iii])))
            
            if self.X is None:
                self.X = data
            else:
                self.X = torch.cat(
                        (self.X, data
                        )
                    )
        self.YAW = np.array( [ [iii] * 4 for iii in range(0, 360, 15)] * len(self.file_list) ).flatten()


        if n_pairs is None:
            self.n_pairs = self.n_data * 5
        else:
            self.n_pairs = int (n_pairs * self.n_data + 1)

        
        self.N = max(self.n_data, self.n_pairs)
        
        
        self.idx_0 = np.random.choice(range(self.n_data), self.n_pairs)
        self.idx_1 = np.random.choice(range(self.n_data), self.n_pairs)
        
        YAW_1 = self.YAW[self.idx_0]
        YAW_2 = self.YAW[self.idx_1]
        
        diff = np.min( [ (YAW_1 - YAW_2) ** 2, (YAW_1 + 360 - YAW_2) ** 2, (YAW_1 - 360 - YAW_2) ** 2 ], 0)
        
        self.S = np.exp(- diff / sigma ** 2)
        self.y = self.YAW

        if not noise is None:
            self.S += np.random.randn(self.n_pairs) * noise
            self.S[self.S < 0] = 0
            self.S[self.S > 1] = 1        
        
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
                
        return self.X[idx].float(), \
               self.X[idx0].float(), \
               self.X[idx1].float(), \
               self.S[S_idx].item()

    