from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

raw_data = pd.read_csv('../data/train_finish.csv', header=0)
data = raw_data.values

imgs = data[0::, 1::]
labels = data[::, 0]

# 选取 2/3 数据作为训练集， 1/3 数据作为测试集
train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)

# 获取一个支持向量机模型
predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
# 把数据丢进去
predictor.fit(train_features[:400], train_labels[:400])
# 预测结果
result = predictor.predict(test_features[:200])
# 准确率估计
score = accuracy_score(test_labels[:200], result)
print ("svm: the accruacy socre is ", score)