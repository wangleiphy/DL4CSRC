# Computation Graphs and Back Propagation

A minimal example of implementing back propagation computation graph with `numpy`.

## How to use

Open this folder in a terminal and type

```bash
$ python main.py
```

This program will

1). check the consistency of gradients with respect to numeric differentiation,

2). download MNIST dataset (hand written digits) from the internet,

3). build a classification network and learn to classify MNIST images.

## Building blocks of a simple classification networks

The computation graph representation of the above network

<img src="../assets/cgraph_mnist_full.png" width="300px" alt="computation graph"/>

## Node definitions
Mathematical definitions of each node, code realizations are contained in `module.py`

<img src="../assets/formulas.png" width="500px" alt="formulas"/>
