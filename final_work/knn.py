import pandas as pd
import numpy as np
import cv2
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_features(trainset):
    features = []

    hog = cv2.HOGDescriptor('../hog.xml')

    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features, (-1, 324))

    return features

def Predict(testset, trainset, train_labels):
    predict = []
    count = 0

    for test_vec in testset:
        print("This is NO." + str(count)) # 作为测试过程输出
        count += 1

        k_list = [] # k个最近邻
        max_index = -1 # 最远点坐标
        max_distance = 0 # 最远点距离

        for i in range(k):
            label = train_labels[i]
            train_vec = trainset[i]
            # 计算两点欧氏距离
            distance = np.linalg.norm(train_vec - test_vec)
            k_list.append((distance, label))
        
        for i in range(k, len(train_labels)):
            label = train_labels[i]
            train_vec = trainset[i]
            # 计算两点欧氏距离
            distance = np.linalg.norm(train_vec - test_vec)
            # 找10个最远点
            if max_index < 0:
                for j in range(k):
                    if max_distance < k_list[j][0]:
                        max_index = j
                        max_distance = k_list[max_index][0]
            
            if distance < max_distance:
                k_list[max_index] = (distance, label)
                max_index = -1
                max_distance = 0
            
        # 表决
        class_all = 10
        class_count = [0 for i in range(class_all)]
        for distance, label in k_list:
            class_count[label] += 1

        max_class = max(class_count) # 最多数表决
        for i in range(class_all):    # 找到最多数表决对应序号
            if max_class == class_count[i]:
                predict.append(i)
                break
    
    return np.array(predict)

k = 10

if __name__ == '__main__':
    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values # 读取训练集

    imgs = data[0::, 1::]
    labels = data[::, 0]
    # print(labels)

    features = get_features(imgs)
    # print(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)
    # 分离训练集和测试集
    # print(train_features, train_labels)

    time_1 = time.time()
    # 预测
    test_predict = Predict(test_features[:2000], train_features[:4000], train_labels[:4000])
    time_2 = time.time()
    print('predicting cost ', time_2 - time_1, 'seconds\n')

    score = accuracy_score(test_labels[:2000], test_predict)
    print("Accuracy is ", score)