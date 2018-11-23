from __future__ import division
import numpy as np

import subprocess
import os
import struct

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

def sanity_checks(check_funcs,check_loss):
    '''
    Function used to check if all module backward properly

    Args:
        check_funcs(list): list of fucntion to check.
        check_loss(list): list of loss fucntion to check.
    '''
    np.random.seed(5)
    for node in check_funcs:
        print('checking %s' % node.__class__.__name__)
        x = np.random.uniform(size=(5, 6))
        gradient_test(node, x)

    # we take special care of loss fucntion node here,
    # it takes an additional parameter p
    for node in check_loss:
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
