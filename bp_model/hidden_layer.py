#coding=utf-8
import numpy as np
from utils.math_utils import utils
class HiddenLayer(object):
    def __init__(self, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.W = utils.randomWeight(fan_out,fan_in)
        self.b = utils.randomWeight(fan_out,1)
        self.delta=[[0] for i in range(fan_out)]
        self.learningrate=0.05
        self.momentumW=utils.zerosWeight(fan_out,fan_in)
        self.momentumb=utils.zerosWeight(fan_out,1)

    def setLearningrate(self, lr):
        self.learningrate = lr

    def output(self,input):
        '''
        :param input:the input should be a (fan_in,1) vector
        :return: (fan_out,1) vector
        '''
        return utils.sigmoid(utils.add(utils.dot(self.W,input),self.b))

    def update(self,deltaW,deltab):
        self.W=utils.minus(self.W,utils.muti(self.learningrate,deltaW))
        self.b=utils.minus(self.b,utils.muti(self.learningrate,deltab))

    def momentumUpdate(self,deltaW,deltab):
        self.W=utils.minus(self.W,utils.muti(self.learningrate,utils.minus(deltaW,self.momentumW)))
        self.b=utils.minus(self.b,utils.muti(self.learningrate,utils.minus(deltab,self.momentumb)))
        self.momentumW=utils.muti(0.9,deltaW)
        self.momentumb=utils.muti(0.9,deltab)



# a=[[0.03337583205846482], [0.233526024754858], [0.057837936455951484], [0.0599413298311716], [0.017846917054514164], [0.10064760528087506], [0.09807616030612953], [0.24958359074015768], [0.1225268705441914], [0.026637732973686212]]
# print (utils.argmax(a))

if __name__=='__main__':
    a = [[1,2,3],[4,5,6],[7,8,9]]
    print (utils.T(utils.T(a)))
    a=[1,2,3]
    print (utils.T(a))
