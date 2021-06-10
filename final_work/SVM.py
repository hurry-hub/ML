import time
import numpy as np
import cvxopt

def loadData(fileName):             # 预处理数据，分为-1和1的二分类
    features = []
    labels = []
    fr = open(fileName, 'r')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        if int(curLine[0]) == 0:
            labels.append(1)
        else:
            labels.append(-1)
        features.append([int(num) / 255 for num in curLine[1:]])
    return features, labels

class SVM(object):
    def __init__(self):
        pass

    def _init_parameters_(self, features, labels):              # 初始化参数
        self.m = len(features[0])   # 获取样本的维度(784)
        self.P = np.identity(self.m + 1)   # 目标函数中的二次部分，我们在最后把b也算作一个变量
        self.P[self.m][self.m] = 0            # 将变量b初始化为0
        self.P = cvxopt.matrix(self.P)   # 转换为矩阵形式
        self.q = cvxopt.matrix([0.0] * (self.m + 1))   # 目标函数中的一次部分

    def train(self, features, labels):
        # 约束条件中的不等式
        self._init_parameters_(features, labels)
        G = []
        for i in range(self.m):
            G.append([-features[j][i] * labels[j] for j in range(len(features))])
        G.append([-labels[i] * 1.0 for i in range(len(labels))])
        G = cvxopt.matrix(G)
        h = cvxopt.matrix([-1.0] * (len(labels)))

        # 将参数传递给solvers.qp，返回最优解
        cs = cvxopt.solvers
        result = cs.qp(self.P, self.q, G, h)
        results = result['x']
        w = results[:-1]
        b = results[-1]
        return w, b

    def _predict_(self, w, b, features):
        if (np.dot(w.T, features) + b) >= 0:
            return 1
        else:
            return -1

    def predict(self, w, b, features, labels):
        # 错误值计数
        errorCnt = 0
        # 循环遍历测试集中的每一个样本
        for i in range(len(features)):
            # 获取预测值
            predict = self._predict_(w, b, features[i])
            # 与答案进行比较
            if predict != labels[i]:
                # 若错误  错误值计数加1
                errorCnt += 1
        # 返回准确率
        return 1 - (errorCnt / len(features))

if __name__ == '__main__':
    time_1 = time.time()
    # 获取训练集
    print('start read train data')
    train_features, train_labels = loadData('../data/mnist_train.csv')
    time_2 = time.time()
    print('read train data cost ', time_2 - time_1, ' seconds', '\n')

    # 获取测试集
    print('start read test data')
    test_features, test_labels = loadData('../data/mnist_test.csv')
    time_3 = time.time()
    print('read test data cost ', time_3 - time_2, ' seconds', '\n')

    # 计算最优解w*,b*3
    print('start train')
    svm = SVM()
    w, b = svm.train(train_features[:6000], train_labels[:6000])
    time_4 = time.time()
    print('train cost ', time_4 - time_3, ' seconds', '\n')

    # 对训练得到的最优解进行测试
    print('start predict')
    accuracy = svm.predict(w, b, test_features[:3000], test_labels[:3000])
    time_5 = time.time()
    # 显示正确率
    print ("The accruacy socre is ", accuracy)
    print('predicting cost ', time_5 - time_4, ' seconds', '\n')
    # 显示用时时长
    print('All cost', time_5 - time_1, ' seconds')