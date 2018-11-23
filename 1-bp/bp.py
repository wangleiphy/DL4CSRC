from __future__ import division
import numpy as np

import subprocess
import os
import struct

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

def numdiff(node, x, var, y_delta, delta, args):
    '''
    numerical differenciation.

    Args:
        node(obj): user defined neural network node.
        x (ndarray): input array.
        delta: the strength of perturbation used in numdiff.
        args: additional arguments for forward function.

    Returns:
        ndarray: gradient computed.
    '''
    var_raveled = var.ravel()

    var_delta_list = []
    for ix in range(len(var_raveled)):
        var_raveled[ix] += delta / 2.
        yplus = node.forward(x, *args)
        var_raveled[ix] -= delta
        yminus = node.forward(x, *args)
        var_delta = ((yplus - yminus) / delta * y_delta).sum()
        var_delta_list.append(var_delta)

        # restore changes
        var_raveled[ix] += delta / 2.
    return np.array(var_delta_list)


def gradient_test(node, x, args=(), delta=0.01, precision=1e-3):
    '''
    perform sanity check for a node,
    raise an assertion error if failed to pass all sanity checks.

    Args:
        node (obj): user defined neural network node.
        x (ndarray): input array.
        args: additional arguments for forward function.
        delta: the strength of perturbation used in numdiff.
        precision: the required precision of gradient (usually introduced by numdiff).
    '''
    y = node.forward(x, *args)
    # y_delta is the gradient passed from the next node, i.e. dL/dy.
    y_delta = np.random.randn(*y.shape)
    x_delta = node.backward(y_delta)

    for var, var_delta in zip([x] + node.parameters, [x_delta] + node.parameters_deltas):
        var_delta_num = numdiff(node, x, var, y_delta, delta, args)
        np.testing.assert_allclose(var_delta_num.reshape(
            *var_delta.shape), var_delta, atol=precision, rtol=precision)

def sanity_checks():
    '''Function used to check if all module backward properly'''
    np.random.seed(5)
    for node in [Linear(6, 4), Sigmoid(), Softmax(), Mean()]:
        print('checking %s' % node.__class__.__name__)
        x = np.random.uniform(size=(5, 6))
        gradient_test(node, x)

    # we take special care of cross entropy node here,
    # it takes an additional parameter p
    node = CrossEntropy()
    print('checking %s' % node.__class__.__name__)
    p = np.random.uniform(0.1, 1, [5, 6])
    p = p / p.sum(axis=-1, keepdims=True)
    x = np.random.uniform(0.1, 1, [5, 6])
    x = x / x.sum(axis=-1, keepdims=True)
    gradient_test(node, x, args=(p,), precision=1e-1)

def load_MNIST():
    '''
    download and unpack MNIST data.

    Returns:
        tuple: length is 4. They are training set data, training set label,
            test set data and test set label.
    '''
    base = "http://yann.lecun.com/exdb/mnist/"
    objects = ['t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte',
               'train-images-idx3-ubyte', 'train-labels-idx1-ubyte']
    end = ".gz"
    path = "data/raw/"
    cmd = ["mkdir", "-p", path]
    subprocess.check_call(cmd)
    print('Downloading MNIST dataset. Please do not stop the program \
during the download. If you do, remove `data` folder and try again.')
    for obj in objects:
        if not os.path.isfile(path + obj):
            cmd = ["wget", base + obj + end, "-P", path]
            subprocess.check_call(cmd)
            cmd = ["gzip", "-d", path + obj + end]
            subprocess.check_call(cmd)

    def unpack(filename):
        '''unpack a single file.'''
        with open(filename, 'rb') as f:
            _, _, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))
                          [0] for d in range(dims))
            data = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
            return data

    # load objects
    data = []
    for name in objects:
        name = path + name
        data.append(unpack(name))
    labels = np.zeros([data[1].shape[0], 10])
    for i, iterm in enumerate(data[1]):
        labels[i][iterm] = 1
    data[1] = labels
    labels = np.zeros([data[3].shape[0], 10])
    for i, iterm in enumerate(data[3]):
        labels[i][iterm] = 1
    data[3] = labels
    return data

def random_draw(data, label, batch_size):
    '''
    random draw a batch of data and label.

    Args:
        data (ndarray): dataset with the first axis the batch dimension.
        label (ndarray): one-hot label for dataset,
            for example, 3 is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        batch_size (int): size of batch, the number of data to draw.

    Returns:
        tuple: length is 2, They are drawed samples from dataset, and labels.
    '''
    perm = np.random.permutation(data.shape[0])
    data_b = data[perm[:batch_size]]
    label_b = label[perm[:batch_size]]
    return data_b.reshape([data_b.shape[0], -1]) / 255.0, label_b

def match_ratio(result, label):
    '''
    the ratio of result matching target.

    Args:
        result(ndarray): result outputed by neural network.
        label(ndarray): the labels from dataset.

    Returns:
        float: math ratio of result and label.
    '''
    label_p = np.argmax(result, axis=1)
    label_t = np.argmax(label, axis=1)
    ratio = np.sum(label_p == label_t) / label_t.shape[0]
    return ratio

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
        ndarray: gradients of output w.r.t inputs
    '''
    y_delta = 1.0
    for node in net[::-1]:
        y_delta = node.backward(y_delta)
    return y_delta

def train():
    '''Training function'''
    np.random.seed(5)
    batch_size = 100
    learning_rate = 0.5
    dim_img = 784
    num_digit = 10
    # an epoch means running through the training set roughly once
    num_epoch = 10
    train_data, train_label, test_data, test_label = load_MNIST()
    num_iteration = len(train_data) // batch_size

    lossfunc = CrossEntropy()
    # define a list as a network, nodes are chained up
    net = [Linear(dim_img, num_digit), Softmax(), lossfunc, Mean()]

    # display test loss before training
    x, label = random_draw(test_data, test_label, 1000)
    result, loss = net_forward(net, lossfunc, x, label)
    print('Before Training.\nTest loss = %.4f, correct rate = %.3f' % (loss, match_ratio(result, label)))

    for epoch in range(num_epoch):
        for j in range(num_iteration):
            x, label = random_draw(train_data, train_label, batch_size)
            result, loss = net_forward(net, lossfunc, x, label)

            net_backward(net)

            # update network parameters
            for node in net:
                for p, p_delta in zip(node.parameters, node.parameters_deltas):
                    p -= learning_rate * p_delta  # stochastic gradient descent

        print("epoch = %d/%d, loss = %.4f, corret rate = %.2f" %
              (epoch, num_epoch, loss, match_ratio(result, label)))

    x, label = random_draw(test_data, test_label, 1000)
    result, loss = net_forward(net, lossfunc, x, label)
    print('After Training.\nTest loss = %.4f, correct rate = %.3f' % (loss, match_ratio(result, label)))


if __name__ == "__main__":
    sanity_checks()
    train()