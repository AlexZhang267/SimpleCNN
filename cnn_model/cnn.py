# coding=utf-8
from __future__ import print_function
from utils.data_loader import load_train_data, load_validation_data
import theano.tensor as T
import theano
from conv_pool_layer import ConnPoolLayer
from full_connect_layer import FullConnectLayer
from softmax_layer import SoftmaxLayer
import numpy as np
import matplotlib.pyplot as plt
import csv


class CNN(object):
    def __init__(self):
        '''
        :param layouts:
        '''
        self.train_data_x, self.train_data_y = load_train_data()
        self.validation_data_x, self.validation_data_y = load_validation_data()
        # print (self.train_data_x.shape,self.train_data_y))
        self.mini_batch_size = 10

        self.validation_data_x_backup = self.validation_data_x
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



        self.learning_rate = 0.1
        self.n_epochs = 100
        self.validateError=[]
        self.trainError=[]
        self.update_num=0
        self.momentum=0
        self.DEBUG = True
        self.current_epoch = 0
        self.error_validate_image=[]
        self.momentum_rate = 0.9

    def train(self):
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')

        batch_size = 10

        layer0_input = x.reshape((self.mini_batch_size, 1, 28, 28))

        # (28-5+1,28-5+1)
        # after pool(12,12)
        # output is (mini_batch_size, 10, 12, 12)
        layer0 = ConnPoolLayer(
            input=layer0_input,
            filter_shape=(8, 1, 5, 5),  # 10filters, 1 feature map, 5 pixels height, 5 pixels width
            image_shape=(self.mini_batch_size, 1, 28, 28),
        )

        # (12-5+1,12-5+1)
        # after pool (4,4)
        # output is (mini_batch_size,5,4,4)
        layer1 = ConnPoolLayer(
            input=layer0.output,
            filter_shape=(32, 8, 5, 5),
            image_shape=(self.mini_batch_size, 8, 12, 12)
        )

        

        #  转化成(mini_batch_size, 5*4*4)
        # output (mini_batch_size, 30)
        layer2_input = layer1.output.flatten(2)
        layer2 = FullConnectLayer(
            input=layer2_input,
            n_in=512,
            n_out=128
        )

        layer3 = SoftmaxLayer(
            input=layer2.output,
            n_in=128,
            n_out=8
        )

        cost = layer3.negative_log_likelihood(y) + 0.008*(T.sum(layer1.W * layer1.W)+ T.sum(layer2.W * layer2.W)+ T.sum(layer3.W * layer3.W))

        params = layer3.params + layer2.params + layer1.params + layer0.params

        # grads = T.grad(cost, params)



        # updates = [
        #     (param_i, param_i + - self.learning_rate * grad_i)
        #     for param_i, grad_i in zip(params, grads)
        #     ]
        updates = self.gradient_updates_momentum(cost,params)

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

        validation_error_model = theano.function(
            inputs=[index],
            outputs=layer3.error_image(y),
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
            print ("validation error rate:%f"%this_validation_loss)
            self.validateError.append(this_validation_loss)


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
            ave_cost/=self.mini_batch_size
            self.trainError.append(ave_cost)
            print("ave cost is %f"%(ave_cost))
            epoch += 1
            if self.DEBUG and epoch%2==0:
                # print("go on? [y/n]")
                str = raw_input("go on? [y/n]")
                if str=='y':
                    pass
                else:
                    self.current_epoch = epoch
                    break
                self.set_params()

        '''
            to record the error image
        '''
        validation_error_image = [validation_error_model(i)
                             for i in range(self.n_validation_batches)]
        error_image=[]
        with open('../dataset/validation_error.csv','w') as f:
            for count in range(len(validation_error_image)):
                pred = validation_error_image[count][0]
                real = validation_error_image[count][1]
                for subcount in range(len(pred)):
                    if pred[subcount] != real[subcount]:
                        e = self.validation_data_x_backup[count*self.mini_batch_size+subcount]
                        e.append(pred[subcount])
                        e.append(real[subcount])
                        if len(e)!=786:
                            raise TypeError('len of error image is wrong %d'%len(e))
                        error_image.append(self.validation_data_x_backup[count*self.mini_batch_size+subcount])

            csvWriter = csv.writer(f)
            csvWriter.writerows(error_image)


    def plotError(self):
        x=np.linspace(1,self.current_epoch,self.current_epoch)
        plt.scatter(x,self.trainError,c='red')
        plt.scatter(x,self.validateError,c='blue')
        plt.show()



    def gradient_updates_momentum(self,cost,params):
        if self.update_num==0:
            self.update_num+=1
            grads = T.grad(cost, params)
            self.momentum = self.momentum_rate * self.learning_rate * np.asarray(grads)
            return  [(param_i, param_i - self.learning_rate * grad_i/self.mini_batch_size)
                for param_i, grad_i in zip(params, grads)]
        else:
            self.update_num+=1
            grads = T.grad(cost,params)
            updates=[
                (param_i, param_i+momentum_i - self.learning_rate * grad_i/self.mini_batch_size)
                for param_i, grad_i,momentum_i in zip(params, grads,self.momentum)
            ]
            self.momentum = self.momentum_rate*(self.momentum - np.asarray(grads))

            return updates
    def set_params(self):
        ch=raw_input("current learning rate is %f, would you set a new learning rate[y/n]"%self.learning_rate)
        if ch=='Y' or ch == 'y':
            lr = raw_input('please set a new learning rate')
            lr = float(lr)
            self.learning_rate=lr

        else:
            pass
        ch = raw_input("current momentum rate is %f, would you set a new momentum rate[y/n]" % self.momentum_rate)
        if ch == 'Y' or ch == 'y':
            lr = raw_input('please set a new momentum rate')
            lr = float(lr)
            self.momentum_rate = lr

        else:
            pass


if __name__ == '__main__':
    cnn=CNN()
    cnn.train()
    cnn.plotError()


