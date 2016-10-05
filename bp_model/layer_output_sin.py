#coding=utf-8
from utils.utils import utils
class SinOutputLayer(object):
    def __init__(self, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.W = utils.randomWeight(fan_out, fan_in)
        self.b = utils.randomWeight(fan_out, 1)
        self.momentumW = utils.zerosWeight(fan_out, fan_in)
        self.momentumb = utils.zerosWeight(fan_out, 1)

        self.delta = [[0] for i in range(fan_out)]
        self.learningrate = 0.05

    def output(self, input):
        '''
        :param input:the input should be a (fan_in,1) vector
        :return: (fan_out,1) vector
        '''
        return utils.add(utils.dot(self.W, input), self.b)

    def setLearningrate(self,lr):
        self.learningrate = lr

    def update(self, deltaW, deltab):
        self.W = utils.minus(self.W, utils.muti(self.learningrate, deltaW))
        self.b = utils.minus(self.b, utils.muti(self.learningrate, deltab))

    def momentumUpdate(self, deltaW, deltab):
        self.W = utils.minus(self.W, utils.muti(self.learningrate, utils.minus(deltaW, self.momentumW)))
        self.b = utils.minus(self.b, utils.muti(self.learningrate, utils.minus(deltab, self.momentumb)))
        self.momentumW = utils.muti(0.9, deltaW)
        self.momentumb = utils.muti(0.9, deltab)
