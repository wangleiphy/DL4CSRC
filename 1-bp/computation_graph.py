from __future__ import division
import numpy as np

class Linear(object):
    '''
    linear node f(x) = xW + b.

    Attributes:
        parameters (list): variables (input nodes) that directly feed into this node, W and b.
        parameters_deltas (list): gradients for parameters.
    '''
    def __init__(self, input_shape, output_shape, mean=0, variance=0.01):
        self.parameters = [mean + variance * np.random.randn(input_shape, output_shape),
                           mean + variance * np.random.randn(output_shape)]
        self.parameters_deltas = [None, None]

    def forward(self, x, *args):
        '''function itself.'''
        self.x = x
        return np.matmul(x, self.parameters[0]) + self.parameters[1]

    def backward(self, delta):
        '''
        Args:
            delta (ndarray): gradient of L with repect to node's output, dL/dy.

        Returns:
            ndarray: gradient of L with respect to node's input, dL/dx
        '''
        self.parameters_deltas[0] = self.x.T.dot(delta)
        self.parameters_deltas[1] = np.sum(delta, 0)
        return delta.dot(self.parameters[0].T)


class F(object):
    '''base class for functions with no parameters.'''
    def __init__(self):
        self.parameters = []
        self.parameters_deltas = []


class Sigmoid(F):
    '''Sigmoid activation function module'''
    def forward(self, x, *args):
        self.x = x
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, delta):
        return delta * ((1 - self.y) * self.y)


class Softmax(F):
    '''Softmax function module'''
    def forward(self, x, *args):
        self.x = x
        xtmp = x - x.max(axis=-1, keepdims=True) # to avoid overflow
        exps = np.exp(xtmp)
        self.y = exps / exps.sum(axis=-1, keepdims=True)
        return self.y

    def backward(self, delta):
        return delta * self.y - self.y * (delta * self.y).sum(axis=-1, keepdims=True)


class CrossEntropy(F):
    '''CrossEntropy function module'''
    def forward(self, x, p, *args):
        # p is target probability. In MNIST dataset, it represents a one-hot label.
        self.p = p
        self.x = x
        y = -p * np.log(np.maximum(x, 1e-15))
        return y.sum(-1)

    def backward(self, delta):
        return -delta[..., None] * 1. / np.maximum(self.x, 1e-15) * self.p


class Mean(F):
    '''Mean function module'''
    def forward(self, x, *args):
        self.x = x
        return x.mean()

    def backward(self, delta):
        return delta * np.ones(self.x.shape) / np.prod(self.x.shape)

def net_forward(net, lossfunc, x, label):
    '''
    forward function for this sequencial network.

    Args:
        net(list): list of layers in neural network.
        lossfunc(F): the loss function of this neural network.
        x(ndarray): the training set.
        label(ndarray): the label of training set.

    Returns:
        tuple: length is 2, They are result outputed by this neural network,
               and loss.
    '''
    for node in net:
        if node is lossfunc:
            result = x
            x = node.forward(x, label)
        else:
            x = node.forward(x)
    return result, x

def net_backward(net):
    '''
    backward function for this sequencial network.

    Args:
        net(list): list of layers in neural network.

    Returns:
        ndarray: gradients of loss w.r.t inputs
    '''
    y_delta = 1.0
    for node in net[::-1]:
        y_delta = node.backward(y_delta)
    return y_delta

