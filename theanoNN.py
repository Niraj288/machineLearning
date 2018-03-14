!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T

class RegressionLayer(object):
    """Class that represents the linear regression, will be the outputlayer
    of the Network"""
    def  __init__(self, input, n_in, learning_rate):
        self.n_in = n_in
        self.learning_rate = learning_rate
        self.input = input

        self.weights = theano.shared(
            value = np.zeros((n_in, 1), dtype = theano.config.floatX),
            name = 'weights',
            borrow = True
        )   

        self.bias = theano.shared(
            value = 0.0,
            name = 'bias'
        )   

        self.regression = T.dot(input, self.weights) + self.bias
        self.params = [self.weights, self.bias]

    def cost_function(self, y):
    	return ((y - self.regression) ** 2).mean()

x = T.dmatrix('x')

reg = r.RegressionLayer(x, 3, 0)

y = theano.shared(value = 0.0, name = "y")

cost = reg.cost_function(y)

T.grad(cost=cost, wrt=reg.weights)