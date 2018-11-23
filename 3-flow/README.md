Here we will learn about Normalizing Flows for variational calculation. The idea is to transform simple base distribution in the latent space (e.g. independent Gaussian which we can sample directly) to complex target distributions in the data space via iterative change-of-varibales. Normalizing Flows are simple yet elegant generative models which demonstrate **representation learning**. 

Some background readings before start:

- Rui Shu's  [Precursor to Normalizing Flows](http://ruishu.io/2018/05/19/change-of-variables/)

- Eric Jang's tutorial  [1](https://blog.evjang.com/2018/01/nf1.html) and [2](https://blog.evjang.com/2018/01/nf2.html)

- OpenAI's [Glow](https://blog.openai.com/glow/)

Here we employ the Monge-Ampère flow introduced in [this paper](https://arxiv.org/abs/1809.10188) for variational calculation of toy target densities. The goal is to minimize the following loss 
$$
\mathcal{L} = \int d x\, q(x) [\ln q(x) + E (x)],
$$
where $q(x)$ is the model density, and $E(x)$ is a given energy function. One can show that the loss function is lower bounded $\mathcal{L} \ge -\ln Z$, where  $Z = \int d x \, e^{-E(x)}$ is the partition function. One will  arrive at the equality only when the variational density matches to the target density $q(x) = e^{-E(x)}/Z$. 

Please play with the code and finish the following tasks 

- [ ] Make a plot of the loss versus training epochs, and compare with exactly computed $-\ln Z$
- [ ] How to make sense of the learned latent space ?  Could you do something fun with it ? 

