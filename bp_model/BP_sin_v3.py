# coding=utf-8
import numpy as np
from hidden_layer import HiddenLayer
from utils.utils import utils
from layer_output_sin import SinOutputLayer
from utils.data_loader import sinnumberloader
from pylab import plot,show
import matplotlib.pyplot as plt
'''
 每次都新创建一堆数据,难以比较其性能,用同一批数据来测试
 为了进一步提高准确性,可以尝试多加一层
 多加层之后的求导的计算可以做优化
 神经元数目
 目前的cost最好到0.00009,也就是差0.01啊,太大了,一定要改
'''


class BP_sin(object):
    def __init__(self):
        x, y = sinnumberloader()

        self.train_x = x[0:2000]
        self.train_y = y[0:2000]

        self.validate_x = x[2000:]
        self.validate_y = y[2000:]

        self.layer0 = HiddenLayer(1, 30)
        # self.layer1 = HiddenLayer(30, 10)
        self.layer2 = SinOutputLayer(30, 1)

        self.learningrate = 0.08
        self.MAX_EPOCH = 1000

        self.layer0.setLearningrate(self.learningrate)
        # self.layer1.setLearningrate(self.learningrate)
        self.layer2.setLearningrate(self.learningrate)

    def train(self):
        data = zip(self.train_x, self.train_y)
        prev_err = 0
        for epoch in range(self.MAX_EPOCH):
            for d in data:
                x = [[d[0]]]
                y = [[d[1]]]

                output0 = self.layer0.output(x)
                # output1 = self.layer1.output(output0)
                output2 = self.layer2.output(output0)
                # print (output1)
                self.backPropagation(x, output0, output2, y)
            cost = self.cost()
            print ("epoch: %d, cost:%f" % (epoch, cost))

            if epoch % 50 == 0 and not epoch == 0:
                self.learningrate/=2
                self.layer0.setLearningrate(self.learningrate)
                self.layer2.setLearningrate(self.learningrate)
            if epoch%10==0:
                if np.abs(cost-prev_err)<=0.000001:
                    break
                else:
                    prev_err=cost
                print (prev_err,cost)

        self.plotError()

    def backPropagation(self, input, output0, output2, y):
        self.layer2.delta = []
        for j in range(len(output2)):
            d = []
            d.append((output2[j][0] - y[j][0]))
            self.layer2.delta.append(d)

        deltaW21 = utils.dot(self.layer2.delta, utils.T(output0))
        deltab21 = self.layer2.delta
        self.layer2.momentumUpdate(deltaW21, deltab21)

        # self.layer1.delta = utils.inner(utils.dot(utils.T(self.layer2.W), self.layer2.delta),
        #                                 [[o[0] * (1 - o[0])] for o in output1])
        #
        # deltaW10 = utils.dot(self.layer1.delta,utils.T(output0))
        # deltab10 = self.layer1.delta
        # self.layer1.update(deltaW10,deltab10)


        self.layer0.delta = utils.inner(utils.dot(utils.T(self.layer2.W), self.layer2.delta),
                                        [[o[0] * (1 - o[0])] for o in output0])

        deltaW00 = utils.dot(self.layer0.delta, utils.T(input))
        deltab00 = self.layer0.delta

        self.layer0.momentumUpdate(deltaW00, deltab00)

    def cost(self):
        data = zip(self.validate_x, self.validate_y)
        cost = 0.
        for d in data:
            x = [[d[0]]]
            y = [[d[1]]]

            output0 = self.layer0.output(x)
            # output1 = self.layer1.output(output0)
            output2 = self.layer2.output(output0)
            cost += (output2[0][0] - y[0][0]) ** 2
        cost = cost / len(self.validate_y)
        return cost

    def test(self):
        str = ""
        while (not str == 'N'):
            str = raw_input("please enter a number")
            number = float(str)
            x = [[number]]
            y = np.sin(number)
            tmp = self.layer0.output(x)
            # output1 = self.layer1.output(tmp)
            output2 = self.layer2.output(tmp)
            print(output2)
            y_pred = output2[0][0]
            print ("pred_num: %f,true value %f, bias is %f" % (y_pred, y, y - y_pred))
    def plotError(self):
        data = zip(self.validate_x, self.validate_y)
        cost = 0.
        xx=[]
        error=[]
        error2=[]
        for d in data:
            x = [[d[0]]]
            y = [[d[1]]]
            output0 = self.layer0.output(x)
            # output1 = self.layer1.output(output0)
            output2 = self.layer2.output(output0)
            cost += (output2[0][0] - y[0][0])
            xx.append(d[0])
            error.append(output2[0][0]-y[0][0])
            error2.append((output2[0][0]-y[0][0])**2)
        # plot(xx,error)
        plt.scatter(xx,error,c='red')
        # plt.scatter(xx,error2,c='blue')
        plt.show()



if __name__ == '__main__':
    model = BP_sin()
    model.train()
    model.test()
