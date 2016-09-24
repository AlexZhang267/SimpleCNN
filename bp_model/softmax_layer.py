#coding=utf-8
from utils.utils import utils
class SoftmaxLayer(object):
    def __init__(self, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.W = utils.randomWeight(fan_out, fan_in)
        self.b = utils.randomWeight(fan_out, 1)
        self.delta=[[0] for i in range(fan_out)]
        self.learningrate=0.05

    def setLearningrate(self, lr):
        self.learningrate = lr

    def output(self, input):
        '''
        :param input:the input should be a (fan_in,1) vector
        :return: (fan_out,1) vector
        '''
        output =utils.exp(utils.add(utils.dot(self.W, input), self.b))
        sum = utils.sum(output)
        return utils.divideNumber(output,sum)

    def update(self, deltaW, deltab):
        self.W=utils.minus(self.W, utils.muti(self.learningrate,deltaW))
        self.b=utils.minus(self.b, utils.muti(self.learningrate,deltab))
