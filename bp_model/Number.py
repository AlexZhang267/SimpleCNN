#coding=utf-8
import random

from utils.data_loader import load_number
from hidden_layer import HiddenLayer
from softmax_layer import SoftmaxLayer
from utils.utils import utils

class BPNN(object):
    def __init__(self):
        self.dataset = load_number()
        self.dataset=[ [float(e) for e in d]for d in self.dataset]
        self.layer0 = HiddenLayer(7, 30)
        # self.layer1 = HiddenLayer(10, 30)
        self.layer2 = SoftmaxLayer(30, 10)

        self.learningrate=0.008

        self.MAX_EPOCH=1000

    def train(self):
        DATASIZE = len(self.dataset)
        for epoch in range(self.MAX_EPOCH):
            error=0.
            random.shuffle(self.dataset)

            self.train_x=[d[0:7] for d in self.dataset]
            self.train_y=[d[-1] for d in self.dataset]

            for i in range(len(self.train_x)):
                self.train_x[i] = [self.train_x[i]]

            for i in range(DATASIZE):
                input_x = utils.T(self.train_x[i])
                input_y = self.train_y[i]
                layer0Out = self.layer0.output(input_x)
                # layer1Out = self.layer1.output(layer0Out)
                layer2Out = self.layer2.output(layer0Out)

                y_pred=utils.argmax(layer2Out)
                # print(input_y,y_pred,layer2Out)


                if not input_y==y_pred:
                    error+=1
                    # self.backPropagation(input_x,layer0Out,layer1Out,layer2Out,input_y)
                    self.backPropagation(input_x,layer0Out,layer2Out,input_y)
                # print ('after bp')
                # layer0Out = self.layer0.output(input_x)
                # layer1Out = self.layer1.output(layer0Out)
                # layer2Out = self.layer2.output(layer1Out)
                #
                # y_pred = utils.argmax(layer2Out)
                # print(input_y, y_pred, layer2Out)

            if (epoch%20==0):
                print("epoch %d error rate %f"%(epoch,(error/DATASIZE)))

            if(epoch%100==0):
                print (self.layer2.W)
            # print("layer2 ",self.layer2.W)
            # print("layer1 ",self.layer1.W)
            # print ("layer0 ",self.layer0.W)
    def backPropagation(self,input,output0,output2,y):
    # def backPropagation(self,input,output0,output1,output2,y):
        '''
        :param output0: the input for layer1
        :param output1: the inout for layer2
        :param output2:
        :param y:
        :return:
        '''
        tmp0=utils.muti(output2[int(y)][0]-1.0,output0)
        tmp1=[float(t[0]) for t in tmp0]
        # print ("tmp1",tmp1)
        deltaW21=[]
        deltab21=[]

        for i in range(len(output2)):
            if not i==y:
                deltaW21.append([0.0 for j in range(len(output0))])
                deltab21.append([0.0])
                self.layer2.delta[i][0]=0.

            else:
                deltaW21.append(tmp1)
                deltab21.append([output2[int(y)][0]-1.0])
                self.layer2.delta[i][0]=output2[int(y)][0]-1.0

        # print (deltaW21)
        self.layer2.update(deltaW21,deltab21)



        # self.layer1.delta = utils.inner(utils.dot(utils.T(self.layer2.W),self.layer2.delta),[[o[0]*(1-o[0])]for o in output1])
        #
        # deltaW10 = utils.dot(self.layer1.delta,utils.T(output0))
        # deltab10 = self.layer1.delta
        #
        # self.layer1.update(deltaW10,deltab10)

        self.layer0.delta = utils.inner(utils.dot(utils.T(self.layer2.W), self.layer2.delta),
                                        [[o[0] * (1 - o[0])] for o in output0])

        deltaW00 = utils.dot(self.layer0.delta, utils.T(input))
        deltab00 = self.layer0.delta

        self.layer0.update(deltaW00, deltab00)


if __name__=="__main__":
    BPNN().train()




