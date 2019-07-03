import os


import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim


import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train = True, n_pairs = None, sigma = 60.,  cross_validate = None, n_fold = 5, validate_set = False,
                 noise = None):
        root = '/pghbio/dbmi/batmanlab/juc91/data/FashionMNIST/'
        
        np.random.seed(0)
        
        trans = transforms.ToTensor()
                                    
        if train:
            self.data = datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
        else:
            self.data = datasets.FashionMNIST(root=root, train=False, transform=trans, download=True)        

        self.n_data = len(self.data)


        if n_pairs is None:
            self.n_pairs = self.n_data * 1
        else:
            self.n_pairs = int (n_pairs * self.n_data + 1)


        self.N = max(self.n_data, self.n_pairs)


        self.idx_0 = np.random.choice(range(self.n_data), self.n_pairs)
        self.idx_1 = np.random.choice(range(self.n_data), self.n_pairs)
        
        self.S = np.zeros(self.n_pairs)
        
        for iii in range(self.n_pairs):
            l_1 = self.data[self.idx_0[iii]][1]
            l_2 = self.data[self.idx_1[iii]][1]

            self.S[iii] = l_1 == l_2

        if not noise is None:
            idx = np.random.choice(self.n_pairs, int( noise * self.n_pairs), replace = False)
            self.S[idx] = np.random.choice(2, idx.shape[0] )

        self.y = np.array([self.data[iii][1] for iii in range(self.n_data)])

        
        
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
                
                
        data = torch.ones([3, 1, 32, 32])
        
        data[0, :, 2:30, 2:30] = 1 - self.data[idx][0]
        data[1, :, 2:30, 2:30] = 1 - self.data[idx0][0]
        data[2, :, 2:30, 2:30] = 1 - self.data[idx1][0]
        
        return data[0, :, :, :], \
               data[1, :, :, :], \
               data[2, :, :, :], \
               self.S[S_idx].item()