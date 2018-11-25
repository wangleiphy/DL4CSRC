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

class MLP(nn.Module):
    def __init__(self, dim, hidden_size, use_z2=True, device='cpu', name=None):
        super(MLP, self).__init__()
        self.device = device
        if name is None:
            self.name = 'MLP'
        else:
            self.name = name

        self.dim = dim
        self.fc1 = nn.Linear(dim, hidden_size, bias=not use_z2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1, bias=False)

        if use_z2:
            self.activation1 = lncosh
            self.activation1_prime = torch.tanh
            self.activation1_prime2 = tanh_prime 
        else:
            self.activation1 = F.softplus
            self.activation1_prime = torch.sigmoid
            self.activation1_prime2 = sigmoid_prime

        self.activation2 = F.softplus
        self.activation2_prime = torch.sigmoid
        self.activation2_prime2 = sigmoid_prime

    def forward(self, x):
        out = self.activation1(self.fc1(x))
        out = self.activation2(self.fc2(out))
        out = self.fc3(out)
        return out.sum(dim=1)

    #def grad(self, x):
    #    out = self.fc1(x)
    #    a2_prime = self.activation1_prime(out)
    #    out = self.fc2(self.activation1(out))
    #    a3_prime = self.activation2_prime(out)

    #    res = torch.mm(a3_prime, torch.diag(self.fc3.weight[0]))
    #    res = torch.mm(res, self.fc2.weight)
    #    res = res*a2_prime
    #    res = torch.mm(res, self.fc1.weight)
    #    return res

    #def laplacian(self, x):
    #    out = self.fc1(x)
    #    a2_prime = self.activation1_prime(out)
    #    a2_prime2 = self.activation1_prime2(out)
    #    out = self.fc2(self.activation1(out))
    #    a3_prime = self.activation2_prime(out)
    #    a3_prime2 = self.activation2_prime2(out)

    #    res1 = torch.mm(a3_prime2, torch.diag(self.fc3.weight[0]))
    #    res = torch.einsum('bj,kj,ji->bki', (a2_prime, self.fc2.weight, self.fc1.weight))
    #    res1 = res1* ((res**2).sum(dim=2)) 

    #    res2 = torch.mm(a3_prime, torch.diag(self.fc3.weight[0]))
    #    res2 = torch.mm(res2, self.fc2.weight)
    #    res2 = res2*a2_prime2
    #    res2 = torch.mm(res2, self.fc1.weight**2)

    #    return res1.sum(dim=1) + res2.sum(dim=1)
    
    def grad(self, x):
        return torch.autograd.grad(self.forward(x), x, grad_outputs=torch.ones(x.shape[0], device=x.device), create_graph=True)[0]

    def laplacian(self, x):
        '''
        Hutchinsons trick for Laplacian (Hessian trace)
        see http://blog.shakirm.com/2015/09/machine-learning-trick-of-the-day-3-hutchinsons-trick/
        '''
        batchsize = x.shape[0]
        z = torch.randn(batchsize, self.dim).to(x.device)
        grad_z = (self.grad(x)*z).sum(dim=1)
        grad2_z = torch.autograd.grad(grad_z, x, grad_outputs=torch.ones(x.shape[0], device=x.device), create_graph=True)[0]
        return (grad2_z * z).sum(dim=1)

class Simple_MLP(nn.Module):
    '''
    Single hidden layer MLP 
    with handcoded grad and laplacian function
    '''
    def __init__(self, dim, hidden_size, use_z2=True, device='cpu', name=None):
        super(Simple_MLP, self).__init__()
        self.device = device
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
        it is simple enough we code it by hand
        '''
        out = self.activation_prime2(self.fc1(x)) 
        out = torch.mm(out, torch.diag(self.fc2.weight[0]))  
        out = torch.mm(out, self.fc1.weight**2)
        return out.sum(dim=1)
    
