#coding=utf-8
from utils.data_loader import load_train_data,load_validation_data
from utils.utils import utils
from hidden_layer import HiddenLayer
from softmax_layer import SoftmaxLayer
class BP_image(object):
    def __init__(self):
        self.train_x,self.train_y = load_train_data()
        self.validate_x,self.validate_y=load_validation_data()

        self.layer0=HiddenLayer(784,100)
        self.layer1=HiddenLayer(100,30)
        self.layer2=SoftmaxLayer(30,10)

        self.learningrate = 0.08
        self.MAX_EPOCH = 1

        self.DEBUG=True

        self.layer0.setLearningrate(self.learningrate)
        self.layer1.setLearningrate(self.learningrate)
        self.layer2.setLearningrate(self.learningrate)

    def train(self):
        count = 0
        data=zip(self.train_x,self.train_y)
        for epoch in range(self.MAX_EPOCH):
            for d in data:
                if self.DEBUG:
                    count+=1
                    if count>5:
                        break
                outputs=self.forward(d)
                self.back(d,outputs)
            self.error()


    def error(self):
        error=0.0
        data = zip(self.validate_x,self.validate_y)
        N = len(data)
        for d in data:
            y=d[1]
            outputs = self.forward(d)
            output2=outputs[0]
            y_pred=utils.argmax(output2)
            if not y_pred==y:
                error+=1

        print("error rate: %f"%(float(error/N)))




    def forward(self, d):
        '''
        :param x:
        :return:

        x 应该是一个list（向量），应该转化成矩阵
        '''
        x=utils.T([d[0]])
        # print x
        output0 = self.layer0.output(x)
        output1 = self.layer1.output(output0)
        output2 = self.layer2.output(output1)
        # print output2
        # print d[1],utils.argmax(output2)
        return[output2,output1,output0]


    def back(self,d,layer):
        x=[d[0]]
        tmp=d[1]

        output2=layer[0]
        output1=layer[1]
        output0=layer[2]

        y=[]
        for i in range(len(output2)):
            if i ==tmp:
                y.append([1.])
            else:
                y.append([0.])

        self.layer2.delta=[[output2[i][0]-y[i][0]] for i in range(len(output2))]

        deltaW21 = utils.dot(self.layer2.delta, utils.T(output1))
        deltab21 = self.layer2.delta
        self.layer2.update(deltaW21, deltab21)


        self.layer1.delta = utils.inner(utils.dot(utils.T(self.layer2.W), self.layer2.delta),
                                        [[o[0] * (1 - o[0])] for o in output1])
        print ("deltaW21",deltaW21)
        deltaW10 = utils.dot(self.layer1.delta, utils.T(output0))
        deltab10 = self.layer1.delta
        self.layer1.update(deltaW10, deltab10)

        print ("deltaW10",deltaW10)


        self.layer0.delta = utils.inner(utils.dot(utils.T(self.layer1.W), self.layer1.delta),
                                        [[o[0] * (1 - o[0])] for o in output0])

        # print("X",len(x))
        # print(len(x[0]))
        # print(len(self.layer0.delta))
        # print(len(self.layer0.delta[0]))

        deltaW00 = utils.dot(self.layer0.delta, x)
        deltab00 = self.layer0.delta
        self.layer0.update(deltaW00, deltab00)








if __name__=="__main__":
    model = BP_image()
    model.train()
