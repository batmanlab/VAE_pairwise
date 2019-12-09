import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np


class Dataset_binary(torch.utils.data.Dataset):
    def __init__(self, train = True, n_pairs = None):
        # MNIST dataset will be downloaded if it is not available.
        
        
        root = '.'
        trans = transforms.ToTensor()
        if train:
            self.data = datasets.MNIST(root=root, train=True, transform=trans, download=True)
        else:
            self.data = datasets.MNIST(root=root, train=False, transform=trans, download=True)        

        self.n_data = len(self.data)
        
        if n_pairs is None:
            self.n_pairs = self.n_data
        else:
            self.n_pairs = int(n_pairs)


        self.N = max(self.n_data, self.n_pairs)

        # Randomly sample binary pairwise labels

        self.idx_0 = np.random.choice(range(self.n_data), self.n_pairs)
        self.idx_1 = np.random.choice(range(self.n_data), self.n_pairs)
        
        self.S = np.zeros(self.n_pairs)
        
        
        # Compute the binary pairwise similarities.
        for iii in range(self.n_pairs):
            l_1 = self.data[self.idx_0[iii]][1]
            l_2 = self.data[self.idx_1[iii]][1]

            self.S[iii] = l_1 == l_2

        self.y = np.array([self.data[iii][1] for iii in range(self.n_data)])
        
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, index):
        # Return four objects.
        # The first is a random sample in the dataset.
        # The second and third are a pair of samples, where its similarities are described using the forth object.
        idx = index % self.n_data
        idx0 = self.idx_0[index % self.n_pairs].item() 
        idx1 = self.idx_1[index % self.n_pairs].item() 
        S_idx = index % self.n_pairs

        # zero-padding. Make it 32 x 32.
        data = torch.ones([3, 1, 32, 32])

        data[0, :, 2:30, 2:30] = 1 - self.data[idx][0]
        data[1, :, 2:30, 2:30] = 1 - self.data[idx0][0]
        data[2, :, 2:30, 2:30] = 1 - self.data[idx1][0]

        return data[0], \
               data[1], \
               data[2], \
               self.S[S_idx].item()
    
class Dataset_real(torch.utils.data.Dataset):
    def __init__(self, train = True, n_pairs = None, step_size = 1.):
        # MNIST dataset will be downloaded if it is not available.
        root = '.'
        trans = transforms.ToTensor()
        if train:
            self.data = datasets.MNIST(root=root, train=True, transform=trans, download=True)
        else:
            self.data = datasets.MNIST(root=root, train=False, transform=trans, download=True)        

        self.n_data = len(self.data)
        
        if n_pairs is None:
            self.n_pairs = self.n_data
        else:
            self.n_pairs = int(n_pairs)
            
        self.step_size = step_size


        self.N = max(self.n_data, self.n_pairs)

        # Randomly sample real-valued pairwise labels

        self.idx_0 = np.random.choice(range(self.n_data), self.n_pairs)
        self.idx_1 = np.random.choice(range(self.n_data), self.n_pairs)
        
        self.S = np.zeros(self.n_pairs)
        
        
        # Compute the real-valued pairwise similarities.
        for iii in range(self.n_pairs):
            l_1 = self.data[self.idx_0[iii]][1]
            l_2 = self.data[self.idx_1[iii]][1]

            self.S[iii] = np.exp( - ( l_1 - l_2 ) ** 2 / self.step_size )

        self.y = np.array([self.data[iii][1] for iii in range(self.n_data)])
        
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, index):
        # Return four objects.
        # The first is a random sample in the dataset.
        # The second and third are a pair of samples, where its similarities are described using the forth object.
                
        idx = index % self.n_data
        idx0 = self.idx_0[index % self.n_pairs].item() 
        idx1 = self.idx_1[index % self.n_pairs].item() 
        S_idx = index % self.n_pairs

        # zero-padding. Make it 32 x 32.
        data = torch.ones([3, 1, 32, 32])

        data[0, :, 2:30, 2:30] = 1 - self.data[idx][0]
        data[1, :, 2:30, 2:30] = 1 - self.data[idx0][0]
        data[2, :, 2:30, 2:30] = 1 - self.data[idx1][0]

        return data[0], \
               data[1], \
               data[2], \
               self.S[S_idx].item()    
