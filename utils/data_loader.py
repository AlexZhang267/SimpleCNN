#coding=utf-8
Debug = True
import random
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


        # read data from file line by line
        # one line must contain 785 elements: 28 * 28 + 1
        d =[int(e) for e in line.split()]

        if not len(d)==785:
            # print d
            # print len(d)
            # print count
            raise TypeError

        # # reformat data to a list: [[raw data], [tag]]
        # d = [d[:-1],d[-1]]
        # data.append(d[:-1])
        # tag.append(d[-1])
        tmp_data.append(d)
    random.shuffle(tmp_data)
    for d in tmp_data:
        data.append(d[:-1])
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
