# coding=utf-8
from utils.data_loader import load_train_data, load_validation_data
import theano.tensor as T
import theano
from conv_pool_layer import ConnPoolLayer
from full_connect_layer import FullConnectLayer
from softmax_layer import SoftmaxLayer
import numpy as np


class CNN(object):
    def __init__(self):
        '''
        :param layouts:
        '''
        self.train_data_x, self.train_data_y = load_train_data()
        self.validation_data_x, self.validation_data_y = load_validation_data()
        # print (self.train_data_x.shape,self.train_data_y))
        self.mini_batch_size = 5

        self.n_train_batches = len(self.train_data_x) // self.mini_batch_size
        self.n_validation_batches = len(self.validation_data_x) // self.mini_batch_size

        self.train_data_x = theano.shared(
            np.asarray(self.train_data_x,dtype=theano.config.floatX),
            borrow=True
        )
        self.train_data_y = theano.shared(
            np.asarray(self.train_data_y,dtype='int32'),
            borrow=True
        )
        self.validation_data_x = theano.shared(
            np.asarray(self.validation_data_x, dtype=theano.config.floatX),
            borrow=True
        )
        self.validation_data_y = theano.shared(
            np.asarray(self.validation_data_y, dtype='int32'),
            borrow=True
        )



        self.learning_rate = 0.05
        self.n_epochs = 100

    def train(self):
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')

        batch_size = 5

        layer0_input = x.reshape((self.mini_batch_size, 1, 28, 28))

        # (28-5+1,28-5+1)
        # after pool(12,12)
        # output is (mini_batch_size, 10, 12, 12)
        layer0 = ConnPoolLayer(
            input=layer0_input,
            filter_shape=(50, 1, 5, 5),  # 10filters, 1 feature map, 5 pixels height, 5 pixels width
            image_shape=(self.mini_batch_size, 1, 28, 28),
        )

        # (12-5+1,12-5+1)
        # after pool (4,4)
        # output is (mini_batch_size,5,4,4)
        layer1 = ConnPoolLayer(
            input=layer0.output,
            filter_shape=(5, 50, 5, 5),
            image_shape=(self.mini_batch_size, 50, 12, 12)
        )

        #  转化成(mini_batch_size, 5*4*4)
        # output (mini_batch_size, 30)
        layer2_input = layer1.output.flatten(2)
        layer2 = FullConnectLayer(
            input=layer2_input,
            n_in=80,
            n_out=30
        )

        layer3 = SoftmaxLayer(
            input=layer2.output,
            n_in=30,
            n_out=8
        )

        cost = layer3.negative_log_likelihood(y)

        params = layer3.params + layer2.params + layer1.params + layer0.params

        grads = T.grad(cost, params)

        updates = [
            (param_i, param_i - self.learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
            ]

        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: self.train_data_x[index * batch_size:(index + 1) * batch_size],
                y: self.train_data_y[index * batch_size:(index + 1) * batch_size]
            }
        )

        validation_model = theano.function(
            inputs=[index],
            outputs=layer3.errors(y),
            givens={
                x: self.validation_data_x[index * batch_size:(index + 1) * batch_size],
                y: self.validation_data_y[index * batch_size:(index + 1) * batch_size]
            }
        )


        print("training")

        epoch = 0
        while (epoch < self.n_epochs):
            print("epoch:%d"%(epoch))

            validation_losses = [validation_model(i)
                                 for i in range(self.n_validation_batches)]
            this_validation_loss = np.mean(validation_losses)
            print ("error rate:%f"%this_validation_loss)


            ave_cost = []
            for mini_batch_index in range(self.n_train_batches):
                cost_ij = train_model(mini_batch_index)
                iter = (epoch) * self.n_train_batches + mini_batch_index
                ave_cost.append(cost_ij)

                if mini_batch_index % 100 == 0:
                    print('training @ iter = %d mini batch index is %d' % (iter,mini_batch_index))
                    # print cost_ij

                # if mini_batch_index>200:
                #     break
            print (ave_cost)
            ave_cost = np.mean(ave_cost)
            print("ave cost is ",ave_cost)
            epoch += 1



if __name__ == '__main__':
    CNN().train()
