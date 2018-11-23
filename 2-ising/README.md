Here we learn about **differentiable scientific programing** using our beloved Ising model as an example. 

Some background readings before start:

- [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767)
- Andrej Karpathy's [article on Software 2.0](https://medium.com/@karpathy/software-2-0-a64152b37c35)

First, [onsager.jl](https://github.com/wangleiphy/DL4CSRC/blob/master/2-ising/onsager.jl) computes the partition function using the Onsager's exact solution of Ising model. Magically, the energy and specific heat automatically follow the "forward mode differentiation". Watch this video to learn about the technique behind it [Automatic Differentiation in 10 minutes with Julia](https://www.youtube.com/watch?v=vAp6nUMrKYg). 

Next, it is actually possible to compute gradient with respect to a whole bunch of nontrivial operations (such as a program).  [trg.py](https://github.com/wangleiphy/DL4CSRC/blob/master/2-ising/trg.py) shows that we can take the gradient through the “[tensor renormalization group](https://arxiv.org/abs/cond-mat/0611687)” (TRG) calculation. Here we use the “reverse mode automatic differentiation” provided by [PyTorch](https://pytorch.org/). A a by product, we also got a GPU version of the TRG almost for free! NB: to have stable gradient through the SVD operation, you need to apply this  [patch](https://github.com/wangleiphy/DL4CSRC/blob/master/2-ising/svd_backward.patch) and recompile PyTorch. 

**Think of what can you do about your research with differentiable programming**

