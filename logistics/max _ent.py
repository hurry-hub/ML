import time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def loadData(fileName):
    data = []
    label = []

    fr = open(fileName, 'r')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        if int(curLine[0]) == 0:
            label.append(1)
        else:
            label.append(0)
        # 二值化
        data.append([int(int(num) > 50) for num in curLine[1:]])

    return data, label

class maxEnt:
    def __init__(self, traindata, trainlabel, testdata, testlabel):
        self.traindata = traindata          # 训练数据集
        self.trainlabel = trainlabel        # 训练标签集
        self.testdata = testdata            # 测试数据集
        self.testlabel = testlabel          # 测试标签集
        self.num = len(traindata[0])     # 特征数量

        self.N = len(traindata)                 # 总训练集长度
        self.n = 0                                  # 训练集中（xi，y）对数量
        self.M = 10000
        self.fixy = self.calculation_fixy()                # 所有(x, y)对出现的次数
        self.w = [0] * self.n                       # Pw(y|x)中的w
        self.xy2idDict, self.id2xyDict = self.createSearchDict()        # (x, y)->id和id->(x, y)的搜索字典
        self.Ep_xy = self.calculation_Ep_xy()               # Ep_xy期望值

    def calculation_Epxy(self):
        # 计算特征函数f(x, y)关于模型P(Y|X)与经验分布P_(X, Y)的期望值 P后带下划线“_”表示P上方的横线
        Epxy = [0] * self.n
        for i in range(self.N):
            # 初始化公式中的P(y|x)列表
            Pwxy = [0] * 2
            Pwxy[0] = self.calculation_Pwy_x(self.traindata[i], 0)
            Pwxy[1] = self.calculation_Pwy_x(self.traindata[i], 1)

            for feature in range(self.num):
                for y in range(2):
                    if (self.traindata[i][feature], y) in self.fixy[feature]:
                        id = self.xy2idDict[feature][(self.traindata[i][feature], y)]
                        Epxy[id] += (1 / self.N) * Pwxy[y]
        return Epxy

    def calculation_Ep_xy(self):
        # 计算特征函数f(x, y)关于经验分布P_(x, y)的期望值 下划线表示P上方的横线，同理Ep_xy中的“_”也表示p上方的横线
        Ep_xy = [0] * self.n
        for feature in range(self.num):
            # 遍历每个特征中的(x, y)对
            for (x, y) in self.fixy[feature]:
                id = self.xy2idDict[feature][(x, y)]
                # 将计算得到的Ep_xy写入对应的位置中
                # fixy中存放所有对在训练集中出现过的次数，除以训练集总长度N就是概率了
                Ep_xy[id] = self.fixy[feature][(x, y)] / self.N
        return Ep_xy

    def createSearchDict(self):
        '''
        创建查询字典，便于寻找对应关系
        xy2idDict：通过(x,y)对找到其id,所有出现过的xy对都有一个id
        id2xyDict：通过id找到对应的(x,y)对
        '''
        xy2idDict = [{} for i in range(self.num)]
        id2xyDict = {}
        index = 0
        for feature in range(self.num):
            for (x, y) in self.fixy[feature]:
                # 将该(x, y)对存入字典中，要注意存入时通过[feature]指定了存入哪个特征内部的字典
                # 同时将index作为该对的id号
                xy2idDict[feature][(x, y)] = index
                id2xyDict[index] = (x, y)
                index += 1
        return xy2idDict, id2xyDict

    def calculation_fixy(self):
        # 计算(x, y)在训练集中出现过的次数
        fixyDict = [defaultdict(int) for i in range(self.num)]
        for i in range(len(self.traindata)):
            for j in range(self.num):
                fixyDict[j][(self.traindata[i][j], self.trainlabel[i])] += 1
        for i in fixyDict:
            self.n += len(i)

        return fixyDict

    def calculation_Pwy_x(self, X, y):
        numerator = 0
        Z = 0
        for i in range(self.num):
            # 如果该(xi,y)对在训练集中出现过
            if (X[i], y) in self.xy2idDict[i]:
                # 在xy->id字典中指定当前特征i，读取其id
                index = self.xy2idDict[i][(X[i], y)]
                numerator += self.w[index]
            # 同时计算其他一种标签y时候的分子，下面的z并不是全部的分母，再加上上式的分子以后才是完整的分母
            if (X[i], 1-y) in self.xy2idDict[i]:
                index = self.xy2idDict[i][(X[i], 1-y)]
                Z += self.w[index]
        numerator = np.exp(numerator)
        Z = np.exp(Z) + numerator
        return numerator / Z

    def maxEntropyTrain(self, iteration = 500):
        # 设置迭代次数寻找最优解
        for i in range(iteration):
            Epxy = self.calculation_Epxy()
            sigmaList = [0] * self.n
            for j in range(self.n):
                # 迭代尺度法
                sigmaList[j] = (1 / self.M) * np.log(self.Ep_xy[j] / Epxy[j])

            self.w = [self.w[i] + sigmaList[i] for i in range(self.n)]
            # 单次迭代结束
            print('iteration:%d' % i)

    def predict_(self, X):
        # 预测标签
        result = [0] * 2
        for i in range(2):
            result[i] = self.calculation_Pwy_x(X, i)
        if result[0] > result[1]:
            return result.index(result[0])
        else:
            return result.index(result[1])

    def predict(self, testData):
        labels = []
        for feature in testData:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels

if __name__ == '__main__':
    start = time.time()

    # 获取训练集及标签
    print('start read transSet')
    trainData, trainLabel = loadData('../data/mnist_train.csv')
    time_1 = time.time()
    print('read trainData cost ', time_1 - start, ' seconds', '\n')
    # 获取测试集及标签
    print('start read testSet')
    testData, testLabel = loadData('../data/mnist_test.csv')
    time_2 = time.time()
    print('read testData cost ', time_2 - time_1, ' seconds', '\n')
    # 初始化最大熵类
    maxEnt = maxEnt(trainData[:200], trainLabel[:200], testData, testLabel)

    # 开始训练
    print('start to train')
    maxEnt.maxEntropyTrain()
    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' seconds', '\n')

    # 开始测试
    print('start to test')
    predictLabel = maxEnt.predict(testData)
    time_4 = time.time()
    print('predicting cost ', time_4 - time_3, ' seconds', '\n')
    accuracy = accuracy_score(testLabel, predictLabel)
    print('the accuracy is: ', accuracy)
    print('all time spending: ', time.time() - start)