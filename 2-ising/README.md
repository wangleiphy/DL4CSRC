# Differentiable Ising solver

Here we learn about **differentiable scientific programing** using our beloved Ising model. 

Some background readings before start:

- [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767)
- Andrej Karpathy's [article on Software 2.0](https://medium.com/@karpathy/software-2-0-a64152b37c35)

First, [onsager.jl](https://github.com/wangleiphy/DL4CSRC/blob/master/2-ising/onsager.jl) computes the free energy using the Onsager's exact solution of 2D classical Ising model. By differentiating the free energy [by hand](https://github.com/wangleiphy/DL4CSRC/blob/master/assets/Excerpt_Moore_Mertens.png) we can obtain physical quantities such as energy and specific heat. Even cooler, the "forward mode automatic differentiation" gives us the same without any hard work! Watch this video to learn about the technique behind it [Automatic Differentiation in 10 minutes with Julia](https://www.youtube.com/watch?v=vAp6nUMrKYg). 

Next, it is actually possible to compute gradient with respect to a whole bunch of nontrivial operations (such as a program).  [trg.py](https://github.com/wangleiphy/DL4CSRC/blob/master/2-ising/trg.py) shows that we can take the gradient through the “[tensor renormalization group](https://arxiv.org/abs/cond-mat/0611687)” (TRG) calculation. Here we use the “reverse mode automatic differentiation” provided by [PyTorch](https://pytorch.org/). A a by product, we also got a GPU version of the TRG almost for free! N.B: To have stable gradient through the SVD operation, you need to apply this  [patch](https://github.com/wangleiphy/DL4CSRC/blob/master/2-ising/svd_backward.patch) and compile PyTorch from source. 

Finally, [tfim.py](https://github.com/wangleiphy/DL4CSRC/blob/master/2-ising/tfim.py) shows a differentiable eigensolver for 1D quantum Ising model. The goal is to learn its model parameters (Ising couplings and transverse fields) so to maximize its ground state overlap with a given target state. It is a fun setup for Hamiltonian engineering. N.B. Again, consider compile from source or install latest build of PyTorch if you see `the derivative for 'symeig' is not implemented` error. They only implemented backward through `torch.symeig` function [very recently](https://github.com/pytorch/pytorch/pull/8586).   

**You see that differentiable programing is a notion which goes beyond deep neural networks. Think about what can you do with this in mind.**
