# coding=utf-8
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
class ConnPoolLayer(object):
    def __init__(self, input, filter_shape, image_shape, poolsize=(2,2)):
        '''
        :param input: input data
        :param filter_shape: (number of filters, number of input feature maps, filter height, filter width)
        :param image_shape:(batch size, number of feature maps, image height, image width)
        :param poolsize:pool size, default is (2,2)
        '''
        self.input = input

        # 不声明dtype无法在GPU上运行
        # borrow 决定深拷贝(False)还是浅拷贝(True)
        # 初始化filter he 每一层filter的bias


        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        rng = np.random.RandomState()
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound,high=W_bound,size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow =True
        )
        self.b = theano.shared(
            np.asarray(
                np.zeros((filter_shape[0],)),
                dtype = theano.config.floatX
            ),
            borrow = True
        )


        # conv_out.shape?
        conv_out = conv2d(
            input=self.input,
            filters=self.W,
            input_shape=image_shape,
            filter_shape=filter_shape
        )

        #pooled_out shape?
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))

        self.params = [self.W, self.b]

        # self.input = input 有什么用
        self.input = input



