# Computation Graphs and Back Propagation

A minimal example of implementing back propagation computation graph with `numpy`.

## How to use

Open this folder in a terminal and type

```bash
$ python computation_graph.py
```

This program will

1). check the consistency of gradients with respect to numeric differentiation,

2). download MNIST dataset (hand written digits) from the internet,

3). build a classification network and learn to classify MNIST images.

## Building blocks of a simple classification networks

The computation graph representation of the above network and the mathematical definitions of each node

<img src="cgraph_mnist_full.png" width="300px" alt="computation graph"/>
<img src="formulas.png" width="500px" alt="formulas"/>
