import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron():
    def __init__(self):
        self.learning_self = 0.00001
        self.max_iteration = 5000
    
    def predict_prove(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)
    
    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)
        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = 2 * labels[index] - 1
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])

            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            for i in range(len(self.w)):
                self.w[i] += self.learning_self * (y * x[i])

    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_prove(x))
        return labels

if __name__ == '__main__':
    init_data = pd.read_csv('data/train_finish.csv', header=0)
    data = init_data.values

    imgs = data[0::, 1::]
    # print(imgs)
    labels = data[::, 0]
    # print(labels)

    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.2, random_state=23323)
    p = Perceptron()
    p.train(train_features, train_labels)

    test_predict = p.predict(test_features)
    score = accuracy_score(test_labels, test_predict)
    print("Accruacy is", score)