import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class MongeAmpereFlow(nn.Module):
    '''
    Monge-Ampere Flow for generative modeling 
    https://arxiv.org/abs/1809.10188
    dx/dt = grad u(x)
    dlnp(x)/dt = -laplacian u(x) 
    '''
    def __init__(self, net, epsilon, Nsteps, device='cpu', name=None, checkpoint=False):
        super(MongeAmpereFlow, self).__init__()
        self.device = device
        if name is None:
            self.name = 'MongeAmpereFlow'
        else:
            self.name = name
        self.net = net 
        self.dim = net.dim
        self.epsilon = epsilon 
        self.Nsteps = Nsteps
        self.checkpoint = checkpoint

    def integrate(self, x, logp, sign=1, epsilon=None, Nsteps=None):
        #default values
        if epsilon is None:
            epsilon = self.epsilon 
        if Nsteps is None:
            Nsteps = self.Nsteps

        #integrate ODE for x and logp(x)
        def ode(x):
            if self.checkpoint:
                return sign*epsilon*checkpoint(self.net.grad, x), -sign*epsilon*checkpoint(self.net.laplacian, x)
            else:
                return sign*epsilon*self.net.grad(x), -sign*epsilon*self.net.laplacian(x)

        #rk4
        for step in range(Nsteps):
            k1_x, k1_logp = ode(x)
            k2_x, k2_logp = ode(x+k1_x/2)
            k3_x, k3_logp = ode(x+k2_x/2)
            k4_x, k4_logp = ode(x+k3_x)

            x = x + (k1_x/6.+k2_x/3. + k3_x/3. +k4_x/6.) 
            logp = logp + (k1_logp/6. + k2_logp/3. + k3_logp/3. + k4_logp/6.)
                
        return x, logp

    def sample(self, batch_size):
        #initial value from Gaussian
        x = torch.randn(batch_size, self.dim, device=self.device, requires_grad=True)
        logp = -0.5 * x.pow(2).add(math.log(2 * math.pi)).sum(1) 
        return self.integrate(x, logp, sign=1)

    def nll(self, x):
        '''
        integrate backwards, thus it returns logp(0) - logp(T)
        '''
        logp = torch.zeros(x.shape[0], device=x.device) 
        x, logp = self.integrate(x, logp, sign=-1)
        return logp + 0.5 * x.pow(2).add(math.log(2 * math.pi)).sum(1)

    def check_reversibility(self, x, logp):
        z, logp_z = self.integrate(x, logp, sign=-1)
        x_back, logp_back = self.integrate(z, logp_z, sign=1)

        x_error = ((x-x_back).abs().sum()) # check reversibility 
        logp_error = ((logp- logp_back).abs().sum())
        return x_error, logp_error

if __name__=='__main__':
    from net import Simple_MLP
    net = Simple_MLP(dim=2, hidden_size = 32)
    model = MongeAmpereFlow(net, epsilon=0.1, Nsteps=100)
    x, logp = model.sample(10)
    print (model.check_reversibility(x, logp))
