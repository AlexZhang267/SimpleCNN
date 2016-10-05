#coding=utf-8
import numpy as np
from hidden_layer import HiddenLayer
from utils.utils import utils
from layer_output_sin import SinOutputLayer


'''
 为了进一步提高准确性,可以尝试多加一层
 多加层之后的求导的计算可以做优化
 神经元数目
 目前的cost最好到0.01,也就是差0.1啊,太大了,一定要改
'''

class BP_sin(object):
    def __init__(self):
        self.train_x = np.random.uniform(low=-np.pi/2,high=np.pi/2,size=1000)
        self.train_y = [np.sin(d) for d in self.train_x]

        self.validate_x = np.random.uniform(low=-np.pi/2,high= np.pi/2,size=100)
        self.validate_y = [np.sin(d) for d in self.validate_x]


        self.layer0 = HiddenLayer(1,50)
        self.layer1 =SinOutputLayer(50,1)

        self.learningrate=0.08
        self.MAX_EPOCH=100

        self.layer0.setLearningrate(self.learningrate)
        self.layer1.setLearningrate(self.learningrate)

    def train(self):
        data = zip(self.train_x, self.train_y)
        for epoch in range(self.MAX_EPOCH):
            for d in data:
                x = [[d[0]]]
                y =[[d[1]]]

                output0 = self.layer0.output(x)
                output1 = self.layer1.output(output0)
                # print (output1)
                self.backPropagation(x,output0,output1,y)
            cost = self.cost()
            print ("epoch: %d, cost:%f"%(epoch,cost))

            if epoch%10==0 and not epoch==0:
                self.layer0.setLearningrate(self.learningrate*10/epoch)
                self.layer1.setLearningrate(self.learningrate*10/epoch)



    def backPropagation(self,input,output0,output1,y):
        self.layer1.delta = []
        for j in range(len(output1)):
            d = []
            d.append((output1[j][0] - y[j][0]))
            self.layer1.delta.append(d)

        deltaW10 = utils.dot(self.layer1.delta, utils.T(output0))
        deltab10 = self.layer1.delta
        self.layer1.update(deltaW10, deltab10)

        self.layer0.delta = utils.inner(utils.dot(utils.T(self.layer1.W), self.layer1.delta),
                                        [[o[0] * (1 - o[0])] for o in output0])

        deltaW00 = utils.dot(self.layer0.delta, utils.T(input))
        deltab00 = self.layer0.delta

        self.layer0.update(deltaW00, deltab00)

    def cost(self):
        data = zip(self.validate_x, self.validate_y)
        cost=0.
        for d in data:
            x = [[d[0]]]
            y = [[d[1]]]

            output0 = self.layer0.output(x)
            output1 = self.layer1.output(output0)
            cost+=(output1[0][0]-y[0][0])**2
        cost = cost/len(self.validate_y)
        return cost

    def test(self):
        str=""
        while(not str=='N'):
            str = raw_input("please enter a number")
            number = float(str)
            x=[[number]]
            y=np.sin(number)
            tmp = self.layer0.output(x)
            output2 = self.layer1.output(tmp)
            print(output2)
            y_pred=output2[0][0]
            print ("pred_num: %f,true value %f, bias is %f"%(y_pred,y,y-y_pred))

if __name__=='__main__':
    model =BP_sin()
    model.train()
    model.test()

