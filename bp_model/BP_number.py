#coding=utf-8
import random

from utils.data_loader import load_number
from hidden_layer import HiddenLayer
from softmax_layer import SoftmaxLayer
from utils.utils import utils


class BP_number(object):
    def __init__(self):
        self.dataset = load_number()
        self.dataset=[ [float(e) for e in d]for d in self.dataset]
        self.learningrate=0.08

        self.layer0 = HiddenLayer(7, 30)
        # self.layer1 = HiddenLayer(10, 30)
        self.layer2 = HiddenLayer(30, 10)

        self.MAX_EPOCH=1000
        self.layer0.setLearningrate(self.learningrate)
        self.layer2.setLearningrate(self.learningrate)


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

                #
                # if not input_y==y_pred:
                #     error+=1
                #     # self.backPropagation(input_x,layer0Out,layer1Out,layer2Out,input_y)
                self.backPropagation(input_x,layer0Out,layer2Out,input_y)

            if(epoch%20==0):
                print("epoch %d error rate %f"%(epoch,(error/DATASIZE)))
            # print("layer1 ",self.layer2.W)
    def backPropagation(self,input,output0,output2,y):
    # def backPropagation(self,input,output0,output1,output2,y):
        '''
        :param output0: the input for layer1
        :param output1: the inout for layer2
        :param output2:
        :param y:
        :return:
        '''
        tmp=[0 for i in range(len(output2))]
        tmp[int(y)]=1

        self.layer2.delta=[]
        for j in range(len(output2)):
            d=[]
            d.append((output2[j][0]-tmp[j])*output2[j][0]*(1-output2[j][0]))
            self.layer2.delta.append(d)


        deltaW21 = utils.dot(self.layer2.delta,utils.T(output0))
        deltab21 = self.layer2.delta
        self.layer2.update(deltaW21,deltab21)

        self.layer0.delta = utils.inner(utils.dot(utils.T(self.layer2.W), self.layer2.delta),
                                        [[o[0] * (1 - o[0])] for o in output0])

        deltaW00 = utils.dot(self.layer0.delta, utils.T(input))
        deltab00 = self.layer0.delta

        self.layer0.update(deltaW00, deltab00)

    def test(self):
        str = raw_input("please enter a number:[1,1,1,1,1,1,1]")
        while(not str=="N"):
            str = raw_input("please enter a number:[1,1,1,1,1,1,1]")
            num = str.split()
            num = [[int(n) for n in num]]
            num = utils.T(num)
            print (num)

            output1 =  self.layer0.output(num)
            output2 = self.layer2.output(output1)
            print(output2)
            print(utils.argmax(output2))


if __name__=='--main__':
    model = BP_number()
    model.train()
    model.test()
