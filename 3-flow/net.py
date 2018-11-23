import math 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

def lncosh(x):
    return -x + F.softplus(2.*x) - math.log(2.)

def tanh_prime(x):
    return 1.-torch.tanh(x)**2

def tanh_prime2(x):
    t = torch.tanh(x)
    return 2.*t*(t*t-1.)

def sigmoid_prime(x):
    s = torch.sigmoid(x)
    return s*(1.-s)

class Simple_MLP(nn.Module):
    '''
    Single hidden layer MLP 
    with handcoded grad and laplacian function
    '''
    def __init__(self, dim, hidden_size, use_z2=True,name=None):
        super(Simple_MLP, self).__init__()
        if name is None:
            self.name = 'Simple_MLP'
        else:
            self.name = name

        self.dim = dim
        self.fc1 = nn.Linear(dim, hidden_size, bias=not use_z2)
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)

        if use_z2:
            self.activation = lncosh
            self.activation_prime = torch.tanh
            self.activation_prime2 = tanh_prime 
        else:
            self.activation = F.softplus
            self.activation_prime = torch.sigmoid
            self.activation_prime2 = sigmoid_prime

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return out.sum(dim=1)

    def grad(self, x):
        '''
        grad u(x)
        '''
        out = self.activation_prime(self.fc1(x)) 
        out = torch.mm(out, torch.diag(self.fc2.weight[0]))  
        out = torch.mm(out, self.fc1.weight)
        return out

    def laplacian(self, x):
        '''
        div \cdot grad u(x)
        '''
        out = self.activation_prime2(self.fc1(x)) 
        out = torch.mm(out, torch.diag(self.fc2.weight[0]))  
        out = torch.mm(out, self.fc1.weight**2)
        return out.sum(dim=1)
    
