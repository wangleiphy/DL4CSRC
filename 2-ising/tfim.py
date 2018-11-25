import torch
import torch.nn as nn
import numpy as np 

def bget(i,p):
    '''
    return the p-th bit of the word i
    '''
    return (i >> p) & 1

def bflip(i,p):
    '''
    return the integer i with the bit at position p flipped: (1->0, 0->1)
    '''
    return i ^(1<<p)

class TFIM(nn.Module):
    def __init__(self, L):
        super(TFIM, self).__init__()

        self.L = L
        self.couplings = nn.Parameter(0.01*torch.randn(L))
        self.fields = nn.Parameter(0.01*torch.randn(L))

    def _buildH(self):
        Nstates = 1 << self.L
        H = torch.zeros(Nstates, Nstates)
        # loop over all basis states
        for i in range(Nstates):
            #diagonal term
            for si in range(self.L):
                H[i,i] += -self.couplings[si]* (2*bget(i, si)-1) * (2*bget(i, si+1)-1)
            
            #off-diagonal term
            for site in range(self.L):
                j = bflip(i, site)
                H[i,j] = -self.fields[site]
        return H

    def overlap(self, psi):
        w, v = torch.symeig(self._buildH(), eigenvectors=True)
        return (v[0, :] * psi).abs().sum()


if __name__=='__main__':
    import sys 
    L = 8  
    model = TFIM(L)

    # a normalized target state 
    target = torch.randn(1<<L)
    target /= target.norm() 

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

    params = list(model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    
    for e in range(100):
        loss = model.overlap(target)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        print (e, loss.item())

 
