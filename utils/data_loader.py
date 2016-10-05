#coding=utf-8
Debug = True
import numpy as np
import random
import csv
def load_train_data():
    train_data = 'dataset/train_data.txt'
    return load_data(train_data)

def load_validation_data():
    validation_data = 'dataset/validation_data.txt'
    return load_data(validation_data)


def load_data(filename):
    try:
        fin = open(filename,'r')
    except IOError:
        fin=open('../'+filename,'r')

    count = 0
    data = []
    tag = []
    tmp_data=[]
    for line in fin:

        if Debug:
            count += 1

        d =[float(e) for e in line.split()]

        if not len(d)==785:
            raise TypeError('pixel length is not 785!')

        tmp_data.append(d)
    random.shuffle(tmp_data)
    for d in tmp_data:
        data.append(d[:-1])
        data=[d/255. for d in data]
        tag.append(d[-1])


    # print len(data)
    return data,tag

def load_number():
    f = open('../dataset/number.txt')
    train_x=[]
    train_y=[]
    dataset=[]
    for line in f:
        line = line.strip().split()
        dataset.append(line)
        # assert len(line)==8
        # train_x.append(line[0:7])
        # train_y.append(line[-1])

    # train_x=[[float(e) for e in row] for row in train_x]
    # train_y=[[float(e) for e in row] for row in train_y]

    return dataset


def initsindata():
    f = open('dataset/sin_number_2.txt','a')
    writer = csv.writer(f)
    '''
    2000 个作为训练机
    500  个作为验证
    '''
    x_trian = np.random.uniform(low=-np.pi*3 / 4, high=np.pi*3/ 4, size=2000)
    y_train = [np.sin(d) for d in x_trian]

    data=zip(x_trian,y_train)
    writer.writerow(('x','y'))
    writer.writerows(data)

    x_validate=np.random.uniform(low=-np.pi/2,high = np.pi/2,size=500)
    y_validate = [np.sin(d) for d in x_validate]
    data_validate=zip(x_validate,y_validate)
    writer.writerows(data_validate)


def sinnumberloader():
    x=[]
    y=[]
    with open('dataset/sin_number_2.txt','r') as csvfile:
        dict_reader = csv.DictReader(csvfile)
        for row in dict_reader:
            x.append(float(row['x']))
            y.append(float(row['y']))

    if not len(x)==len(y):
        raise AttributeError('x.length is not equal to y.length')
    return x,y

