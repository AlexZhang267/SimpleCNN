# coding=utf-8
import theano.tensor as T
import theano
import numpy as np


class FullConnectLayer(object):
    def __init__(self, input, n_in, n_out):
        rng = np.random.RandomState()
        self.input = input
        W_bound = np.sqrt(6./(n_in+n_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)),
                dtype=T.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            np.asarray(
                rng.uniform(
                    low=0,
                    high=1,
                    size=(n_out,)
                ),
                dtype=T.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.output = T.nnet.sigmoid(T.dot(self.input, self.W) + self.b)
        self.params = [self.W, self.b]
