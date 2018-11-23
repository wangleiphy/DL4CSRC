from __future__ import division
import numpy as np

from etc import sanity_checks, load_MNIST, random_draw, match_ratio
from module import Linear, Sigmoid, Softmax, Softmax, CrossEntropy, Mean, net_backward, net_forward

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
    sanity_checks([Linear(6, 4), Sigmoid(), Softmax(), Mean()],[CrossEntropy()])
    train()