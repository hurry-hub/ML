import numpy as np
from math import log

def loadData():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels

def calc_entropy(datasets, index=-1):
    label_count = {}
    for dataset in datasets:
        label =  dataset[index]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    entropy = -sum([(p/len(datasets))*log(p/len(datasets),2) for p in label_count.values()])
    return entropy

def calc_conditional_entropy(datasets, index = 0):
    feature_data = {}
    for dataset in datasets:
        feature = dataset[index]
        if feature not in feature_data:
            feature_data[feature] = []
        feature_data[feature].append(dataset)
    condEntropy = sum([(len(p)/len(datasets))*calc_entropy(p) for p in feature_data.values()])
    return condEntropy

def info_gain(entropy, condEntropy):
    return entropy - condEntropy

def info_gain_ratio(c_info_gain, c_entropy):
    return 0 if c_info_gain == 0 else c_info_gain / c_entropy

def info_gain_train_childTree(datasets, labels):
    entropy = calc_entropy(datasets)
    features = []
    for index in range(len(datasets[0])-1):
        condEntropy = calc_conditional_entropy(datasets, index)
        c_info_gain = info_gain(entropy, condEntropy)
        c_entropy = calc_entropy(datasets, index)
        c_info_gain_ratio = info_gain_ratio(c_info_gain, c_entropy)
        features.append((index, c_info_gain_ratio))
        print("特征({})的信息增益比为{:.3f}".format(labels[index], c_info_gain_ratio))
    best_feature = max(features, key=lambda x: x[-1])
    print("特征({})的信息增益比最大，选择为当前节点特征".format(labels[best_feature[0]]))
    return best_feature

def info_gain_train(datasets, labels):
    label_count = {}
    for dataset in datasets:
        label = dataset[-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    if len(label_count.keys()) == 1:
        key = list(label_count.keys())[0]
        print("此时类别均为{}".format(key))
        return
    best_feature = info_gain_train_childTree(datasets, labels)

    feature_data = {}
    for dataset in datasets:
        feature = dataset[best_feature[0]]
        if feature not in feature_data:
            feature_data[feature] = []
        feature_data[feature].append(dataset)

    for data in zip(feature_data.keys(), feature_data.values()):
        print("当{}为{}".format(labels[best_feature[0]], data[0]))
        info_gain_train(data[1], labels)


if __name__ == "__main__":
    datasets, labels = loadData()
    info_gain_train(datasets, labels)