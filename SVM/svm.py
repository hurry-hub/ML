import time
import logging
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class SVM(object):

    def __init__(self, kernel='gauss',epsilon = 0.001):
        self.kernel = kernel
        self.epsilon = epsilon
        self.sigma = 10

    def _init_parameters(self, features, labels):  # 初始化
        self.X = features
        self.Y = labels

        self.b = 0.0
        self.n = len(features[0])
        # print(self.n)
        self.N = len(features)
        # print(self.N)
        self.alpha = [0.0] * self.N
        self.E = [self._E_(i) for i in range(self.N)]

        self.C = 1000
        self.Max_Interation = 500

    def _satisfy_KKT(self, i):
        ygx = self.Y[i] * self._g_(i)
        if abs(self.alpha[i])<self.epsilon:
            return ygx > 1 or ygx == 1
        elif abs(self.alpha[i]-self.C)<self.epsilon:
            return ygx < 1 or ygx == 1
        else:
            return abs(ygx-1) < self.epsilon

    def is_stop(self):
        for i in range(self.N):
            satisfy = self._satisfy_KKT(i)

            if not satisfy:
                return False
        return True

    def _select_two_parameters(self):           # 按照书上7.4.2选择两个变量
        index_list = [i for i in range(self.N)]

        i1_list_1 = list(filter(lambda i: self.alpha[i] > 0 and self.alpha[i] < self.C, index_list))        # 第一个变量
        i1_list_2 = list(set(index_list) - set(i1_list_1))                                                  # 第二个变量

        i1_list = i1_list_1
        i1_list.extend(i1_list_2)

        for i in i1_list:
            if self._satisfy_KKT(i):
                continue

            E1 = self.E[i]
            max_ = (0, 0)

            for j in index_list:
                if i == j:
                    continue

                E2 = self.E[j]
                if abs(E1 - E2) > max_[0]:
                    max_ = (abs(E1 - E2), j)

            return i, max_[1]

    def kernel_(self, x1, x2):      # 核函数
        if self.kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])              # 线性核
        if self.kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)])+1)**3       # 多项式核
        if self.kernel == 'gauss':
            return np.exp(-sum([(x1[k] - x2[k]) ** 2 for k in range(self.n)]) / (2 * self.sigma ** 2))
        print ('没有定义核函数')
        return 0

    def _g_(self, i):       # 公式(7.104)
        result = self.b

        for j in range(self.N):
            result += self.alpha[j] * self.Y[j] * self.kernel_(self.X[i], self.X[j])

        return result

    def _E_(self, i):       # 公式(7.105)
        return self._g_(i) - self.Y[i]

    def try_E(self,i):
        result = self.b-self.Y[i]
        for j in range(self.N):
            if self.alpha[j]<0 or self.alpha[j]>self.C:
                continue
            result += self.Y[j]*self.alpha[j]*self.kernel_(self.X[i],self.X[j])
        return result


    def train(self, features, labels):

        self._init_parameters(features, labels)

        for times in range(self.Max_Interation):

            logging.debug('iterater %d' % times)

            i1, i2 = self._select_two_parameters()

            L = max(0, self.alpha[i2] - self.alpha[i1])                     # 对角线段端点的界
            H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i2] + self.alpha[i1] - self.C)
                H = min(self.C, self.alpha[i2] + self.alpha[i1])

            E1 = self.E[i1]                                                 # 预测值与真实数据的差
            E2 = self.E[i2]
            eta = self.kernel_(self.X[i1], self.X[i1]) + self.kernel_(self.X[i2], self.X[i2]) - 2 * self.kernel_(self.X[i1], self.X[i2])     # 公式(7.107)

            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta        # 公式(7.106)
            # 公式(7.108)
            alph2_new = 0
            if alpha2_new_unc > H:
                alph2_new = H
            elif alpha2_new_unc < L:
                alph2_new = L
            else:
                alph2_new = alpha2_new_unc
            # 公式(7.109)
            alph1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alph2_new)

            # 公式(7.115) 及 公式(7.116)
            b_new = 0
            b1_new = -E1 - self.Y[i1] * self.kernel_(self.X[i1], self.X[i1]) * (alph1_new - self.alpha[i1]) \
                     - self.Y[i2] * self.kernel_(self.X[i2], self.X[i1]) * (alph2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel_(self.X[i1], self.X[i2]) * (alph1_new - self.alpha[i1]) \
                     - self.Y[i2] * self.kernel_(self.X[i2], self.X[i2]) * (alph2_new - self.alpha[i2]) + self.b

            if alph1_new > 0 and alph1_new < self.C:                # 得到b_new
                b_new = b1_new
            elif alph2_new > 0 and alph2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            self.alpha[i1] = alph1_new
            self.alpha[i2] = alph2_new
            self.b = b_new

            self.E[i1] = self._E_(i1)
            self.E[i2] = self._E_(i2)


    def _predict_(self,feature):
        result = self.b
        for i in range(self.N):
            result += self.alpha[i]*self.Y[i]*self.kernel_(feature,self.X[i])
        if result > 0:
            return 1
        return -1

    def predict(self, features):
        results = []
        for feature in features:
            results.append(self._predict_(feature))
        return results


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    print ('Start read data')

    time_1 = time.time()

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    '''train_features, train_labels, test_features, test_labels = generate.generate_dataset(2000,visualization=False)
    print(train_features)
    print(train_labels)'''

    raw_data = pd.read_csv('../data/train_finish.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.3, random_state=23323)
    train_features = train_features[:400]
    train_labels = train_labels[:400]
    test_features = test_features[:200]
    test_labels = test_labels[:200]

    time_2 = time.time()
    print ('read data cost ',time_2 - time_1,' second','\n')

    print ('Start training')
    svm = SVM()
    svm.train(train_features, train_labels)

    time_3 = time.time()
    print ('training cost ',time_3 - time_2,' second','\n')

    print ('Start predicting')
    test_predict = svm.predict(test_features)
    time_4 = time.time()
    print ('predicting cost ',time_4 - time_3,' second','\n')

    score = accuracy_score(test_labels,test_predict)
    print ("svm: the accruacy socre is ", score)

