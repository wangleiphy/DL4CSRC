Here we will learn about Normalizing Flows for variational calculation. Some background readings before start:

- Rui Shu's  Precursor [http://ruishu.io/2018/05/19/change-of-variables/](http://ruishu.io/2018/05/19/change-of-variables/)

- Eric Jang's tutorial  [1](https://blog.evjang.com/2018/01/nf1.html) and [2](https://blog.evjang.com/2018/01/nf2.html)

- OpenAI's [Glow](https://blog.openai.com/glow/)

In the ... we use the Monge-Ampère flow introduced in [this paper](https://arxiv.org/abs/1809.10188) for variational calculation of toy target densities. The goal is to minimize the following loss 
$$
\mathcal{L} = \int d x\, q(x) [\ln q(x) + E (x)],
$$
where $q(x)$ is the model density, and $E(x)$ is a given energy function. One can show that the loss function is lower bounded $\mathcal{L} \ge -\ln Z$, where  $Z = \int d x \, e^{-E(x)}$ is the partition function. One arrives at the equality only when the variational density matches to the target density $q(x) = e^{-E(x)}/Z$. 

Play with the code, and finish the following tasks 

- [ ] Make a plot of the loss versus training epochs, and compare with exactly computed $-\ln Z$
- [ ] Think about how to make sense of the learned latent space. 
- [ ] Could you do something fun with the learned latent space. ? 

