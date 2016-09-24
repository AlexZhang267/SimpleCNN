# coding=utf-8
import random
import math


class utils(object):
    @classmethod
    def randomWeight(cls, height, width):
        W_bound = math.sqrt(6.0 / (height + width))
        if height*width>1000:
            W_bound/=1000
        return [[random.uniform(-W_bound, W_bound) for w in range(width)] for h in range(height)]

    '''
    仅仅支持二维矩阵乘以二维矩阵
    '''

    @classmethod
    def dot(cls, var1, var2):

        if not len(var1) > 0:
            print(var1)
            print(len(var1))
            raise TypeError

        if not len(var1[0]) == len(var2):
            print(len(var1[0]))
            print(len(var2))
            raise TypeError

        out = []
        for var3 in var1:
            tmp = [0 for i in range(len(var2[0]))]
            for j in range(len(var2)):
                for k in range(len(var2[j])):
                    tmp[k] += var3[j] * var2[j][k]
            out.append(tmp)

        assert len(out) == len(var1)
        assert len(out[0]) == len(var2[0])
        return out

    @classmethod
    def add(cls, var1, var2):
        assert len(var1) == len(var2) and len(var1) > 0
        assert len(var1[0]) == len(var2[0]) and len(var1[0]) > 0

        return [[var1[i][j] + var2[i][j] for j in range(len(var1[0]))] for i in range(len(var1))]

    @classmethod
    def minus(cls, var1, var2):
        assert len(var1) == len(var2) and len(var1) > 0
        assert len(var1[0]) == len(var2[0]) and len(var1[0]) > 0
        return [[var1[i][j] - var2[i][j] for j in range(len(var1[0]))] for i in range(len(var1))]

    @classmethod
    def sigmoid(cls, input):
        '''
        :param input: 矩阵
        :return:
        '''

        return [[cls.__sigmoid(x) for x in var1] for var1 in input]

    @classmethod
    def __sigmoid(cls, x):
        '''
        :param input: 一个数
        :return:
        '''
        try:
            return 1. / (1. + math.exp(-x))
        except OverflowError:
            print x

    '''
    对x内所有元素加和
    '''

    @classmethod
    def sum(cls, x):
        sum = 0
        for a in x:
            for aa in a:
                sum += aa
        return sum

    @classmethod
    def exp(cls, x):
        return [[math.exp(var2) for var2 in var1] for var1 in x]

    @classmethod
    def divideMatrix(cls, x, y):
        assert len(x) == len(y)
        return [[float(x[i][j]) / y[i][j] for j in range(len(x[0]))] for i in range(len(x))]

    @classmethod
    def divideNumber(cls, x, y):
        return [[float(x[i][j]) / y for j in range(len(x[0]))] for i in range(len(x))]

    @classmethod
    def T(cls, x):
        H = len(x)
        if not type(x[0]) == list:
            W = 1
        else:
            W = len(x[0])
        T = []
        # if W>1:
        for w in range(W):
            line = []
            for r in range(H):
                line.append(x[r][w])
            T.append(line)

            # else:
            #     line=[]
            #     for r in range(H):
            #         line.append(x[r][0])
            #     print (line)
            #     T.append(line)

        return T


    @classmethod
    def argmax(cls, x):
        index = 0
        for i in range(len(x)):
            if x[i][0] > x[index][0]:
                index = i

        return index


    @classmethod
    def muti(cls, number, matrix):
        return [[float(number) * float(e) for e in row] for row in matrix]


    @classmethod
    def inner(cls, x, y):
        return [[x[i][j] * y[i][j] for j in range(len(x[0]))] for i in range(len(x))]
