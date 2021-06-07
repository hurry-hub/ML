import cv2
import time
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Sign(object):                             # 分类器
    def __init__(self, features, labels, w):
        self.X = list(features)               # 训练数据特征
        self.Y = list(labels)                 # 训练数据的标签
        self.N = len(list(labels))            # 训练数据大小

        self.w = w                      # 训练数据权值分布
        self.indexes = [0, 1, 2]          # 阈值轴可选范围

    def _less_than_(self):        # 寻找(x<v y=1)情况下的最优v
        index = -1
        error_score = 1000000

        for i in self.indexes:
            score = 0
            for j in range(self.N):
                val = -1
                # print(self.X)
                if self.X[j] < i:
                    val = 1
                if val * self.Y[j] < 0:
                    score += self.w[j]

            if score < error_score:
                index = i
                error_score = score

        return index, error_score

    def _more_than_(self):            # 寻找(x>v y=1)情况下的最优v
        index = -1
        error_score = 1000000

        for i in self.indexes:
            score = 0
            for j in range(self.N):
                val = 1
                if self.X[j] < i:
                    val = -1
                if val * self.Y[j] < 0:
                    score += self.w[j]

            if score < error_score:
                index = i
                error_score = score

        return index, error_score

    def train(self):
        less_index, less_score = self._less_than_()
        more_index, more_score = self._more_than_()

        if less_score < more_score:
            self.is_less = True
            self.index = less_index
            return less_score
        else:
            self.is_less = False
            self.index = more_index
            return more_score

    def predict(self, feature):
        if self.is_less > 0:
            if feature < self.index:
                return 1.0
            else:
                return -1.0
        else:
            if feature < self.index:
                return -1.0
            else:
                return 1.0

class AdaBoost(object):
    def __init__(self):
        pass

    def _init_parameters_(self, features, labels):
        self.X = list(features)                           # 训练集特征
        self.Y = list(labels)                             # 训练集标签

        self.n = len(features[0])                   # 特征维度
        self.N = len(features)                      # 训练集大小
        self.M = 30                                 # 分类器数目

        self.w = [1.0 / self.N] * self.N                # 训练集的权值分布
        self.alpha = []                             # 分类器系数  公式8.2
        self.classifier = []                        # (维度，分类器)，针对当前维度的分类器

    def _w_(self, index, classifier, i):       # 公式8.4 不包括Zm
        return self.w[i]*np.exp(-self.alpha[-1]*self.Y[i]*classifier.predict(self.X[i][index]))

    def _Z_(self, index, classifier):         # 公式8.5
        Z = 0
        for i in range(self.N):
            Z += self._w_(index, classifier, i)

        return Z

    def train(self, features, labels):
        self._init_parameters_(features, labels)

        for times in range(self.M):
            print('iterater %d' % times)
            time1 = time.time()

            best_classifier = (100000, None, None)        # (误差率,针对的特征，分类器)
            for i in range(self.n):
                features = map(lambda x : x[i], self.X)        # 0 <-> 1 -> -1 <-> 1
                classifier = Sign(features, self.Y, self.w)
                error_score = classifier.train()
                if error_score < best_classifier[0]:
                    best_classifier = (error_score, i, classifier)

            em = best_classifier[0]
            time2 = time.time()
            print ('总运行时间:%s' % str(time2-time1))

            if em == 0:
                self.alpha.append(100)
            else:
                self.alpha.append(0.5 * math.log((1 - em) / em))
            self.classifier.append(best_classifier[1:])
            Z = self._Z_(best_classifier[1], best_classifier[-1])

            # 计算训练集权值分布 8.4
            for i in range(self.N):
                self.w[i] = self._w_(best_classifier[1], best_classifier[2],i)/Z

    def _predict_(self, feature):
        result = 0.0
        for i in range(self.M):
            index = self.classifier[i][0]
            classifier = self.classifier[i][1]
            result += self.alpha[i]*classifier.predict(feature[index])
        if result > 0:
            return 1
        return -1

    def predict(self, features):
        results = []
        for feature in features:
            results.append(self._predict_(feature))
        return results

# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY_INV,cv_img)
    return cv_img

def binaryzation_features(trainset):
    features = []
    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        img_b = binaryzation(cv_img)
        features.append(img_b)
    features = np.array(features)
    features = np.reshape(features,(-1,784))

    return features

if __name__ == '__main__':
    print ('Start read data')
    time_1 = time.time()
    raw_data = pd.read_csv('../data/train_finish.csv', header=0)
    data = raw_data.values
    imgs = data[0::, 1::]
    labels = data[::, 0]

    features = binaryzation_features(imgs)              # 二值化
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.5, random_state=0)
    time_2 = time.time()
    print ('read data cost ', time_2 - time_1, ' seconds', '\n')

    print ('Start training')
    train_labels = map(lambda x : 2 * x - 1, train_labels[:400])
    ada = AdaBoost()
    ada.train(train_features[:400], train_labels)
    time_3 = time.time()
    print ('training cost ', time_3 - time_2, ' seconds', '\n')

    print ('Start predicting')
    test_predict = ada.predict(test_features[:400])
    time_4 = time.time()
    print ('predicting cost ', time_4 - time_3, ' seconds', '\n')
    test_labels = map(lambda x : 2 * x - 1, test_labels[:400])
    score = accuracy_score(list(test_labels), test_predict)
    print ("The accruacy socre is ", score)
