#encoding=utf-8

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY_INV,cv_img) # 将大于50的置0，小于50的置1
    return cv_img

def Train(trainset,train_labels):
    prior_pro = np.zeros(class_num)                         # 先验概率
    cond_pro = np.zeros((class_num,feature_len,2))   # 条件概率

    # 计算先验概率及条件概率
    for i in range(len(train_labels)):
        img = binaryzation(trainset[i])     # 图片二值化
        label = train_labels[i]
        # 得到处理后的训练集以及标签
        prior_pro[label] += 1

        for j in range(feature_len):
            cond_pro[label][j][img[j]] += 1

    # 将概率归到[1.10001]
    for i in range(class_num):
        for j in range(feature_len):

            # 经过二值化后图像只有0，1两种取值，否则得到的矩阵过大无法运行
            p0 = cond_pro[i][j][0]
            p1 = cond_pro[i][j][1]

            # 计算0，1像素点对应的条件概率
            pro_0 = (float(p0)/float(p0+p1))*10000 + 1     # 将得到的概率放大，同时+1防止分母为0（拉普拉斯平滑）
            pro_1 = (float(p1)/float(p0+p1))*10000 + 1

            cond_pro[i][j][0] = pro_0
            cond_pro[i][j][1] = pro_1

    return prior_pro,cond_pro

# 计算概率
def calculate_pro(img,label):
    pro = int(prior_pro[label])

    for i in range(len(img)):
        pro *= int(cond_pro[label][i][img[i]]) # 连乘得到条件独立性假设

    return pro

def Predict(testset,prior_pro,cond_pro):
    predict = []

    for img in testset:

        # 图像二值化
        img = binaryzation(img)

        max_label = 0
        max_pro = calculate_pro(img,0)

        for j in range(1,10):
            pro = calculate_pro(img,j)

            if max_pro < pro:
                max_label = j
                max_pro = pro
            # 将后验概率最大的类输出作为预测的类

        predict.append(max_label)

    return np.array(predict)


class_num = 10
feature_len = 784

if __name__ == '__main__':

    print ('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train.csv',header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print ('read data cost ',time_2 - time_1,' seconds\n')

    print ('Start training')
    prior_pro,cond_pro = Train(train_features,train_labels)
    time_3 = time.time()
    print ('training cost', time_3 - time_2, 'seconds\n')

    print ('Start predicting')
    test_predict = Predict(test_features,prior_pro,cond_pro)
    time_4 = time.time()
    print ('predicting cost ',time_4 - time_3,' seconds\n')

    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)
