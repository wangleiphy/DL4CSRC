import torch 
import torch.nn as nn

import numpy as np 
import matplotlib.pyplot as plt 

from net import Simple_MLP, MLP
from flow import MongeAmpereFlow

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-cuda", type=int, default=-1, help="use GPU")
    parser.add_argument("-net", default='Simple_MLP', help="network architecture")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))

    xlimits=[-4, 4]
    ylimits=[-4, 4]
    numticks=21
    x = np.linspace(*xlimits, num=numticks, dtype=np.float32)
    y = np.linspace(*ylimits, num=numticks, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    xy = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
    xy = torch.from_numpy(xy).contiguous().to(device)

    # Set up plotting code
    def plot_isocontours(ax, func, alpha=1.0):
        zs = np.exp(func(xy).cpu().detach().numpy())
        Z = zs.reshape(X.shape)
        plt.contour(X, Y, Z, alpha=alpha)
        ax.set_yticks([])
        ax.set_xticks([])
        plt.xlim(xlimits)
        plt.ylim(ylimits)

    from objectives import Ring2D
    target = Ring2D()
    target.to(device)

    epsilon = 0.1 
    Nsteps = 50
    batch_size = 1024

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    

    if (args.net=='Simple_MLP'):
        net = Simple_MLP(dim=2, hidden_size = 32, device = device)
    elif (args.net=='MLP'):
        net = MLP(dim=2, hidden_size = 32, device = device)
    else:
        print ('what net ?', args.net)
        sys.exit(1)

    model = MongeAmpereFlow(net, epsilon, Nsteps, device=device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

    params = list(model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    
    np_losses = []
    for e in range(100):
        x, logp = model.sample(batch_size)
        loss = logp.mean() - target(x).mean() 
        
        model.zero_grad()
        loss.backward()
        optimizer.step()

        print (e, loss.item())
        np_losses.append([loss.item()])

        plt.cla()
        plot_isocontours(ax, target, alpha=0.5)
        plot_isocontours(ax, model.net) # Breiner potential 

        samples = x.cpu().detach().numpy()
        plt.plot(samples[:, 0], samples[:,1],'o', alpha=0.8)

        plt.draw()
        plt.pause(0.01)

    np_losses = np.array(np_losses)
    fig = plt.figure(figsize=(8,8), facecolor='white')
    plt.ioff()
    plt.plot(np_losses)
    plt.show()
