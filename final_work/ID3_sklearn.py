from sklearn import tree
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

time_1 = time.time()
print('start read data')
raw_data = pd.read_csv('../data/train_finish.csv', header=0)
data = raw_data.values
imgs = data[0::, 1::]
labels = data[::, 0]
# 选取 2/3 数据作为训练集， 1/3 数据作为测试集
train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)
time_2 = time.time()
print('read data cost', time_2 - time_1, ' seconds\n')

# 获取一个决策树模型
clf = tree.DecisionTreeClassifier(criterion="entropy")
# 把数据带入
print('start predict')
time_3 = time.time()
clf.fit(train_features[:6000], train_labels[:6000])
# 预测结果
result = clf.predict(test_features[:3000])
# 准确率估计
score = accuracy_score(test_labels[:3000], result)
print ("tree: the accruacy socre is ", score)
time_4 = time.time()
print('predict data cost', time_4 - time_3, ' seconds')
print('all cost', time_4 - time_1, ' seconds')