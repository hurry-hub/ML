import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

x = np.array([[1, 6.5], [5, 2], [2.6, 1.4], [4, 1], [1.3, 2.1], [4.5, 2.4], [3.2, 5.4], [0.3, 3.7], [1.4, 5.6], [4.6, 1],
              [5, 6], [3, 3], [1.7, 4.7], [4.2, 0.4]])  # 14个点

y = np.array([ 0,  1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]) # 类别

x_min, x_max = 0, 7
y_min, y_max = 0, 7 #设置图片边界

cmap_light = ListedColormap(['#FFFFFF', '#0d3462']) # 设置颜色

h = .01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

knn = KNeighborsClassifier(n_neighbors=2) # n_neighbors即为k
knn.fit(x, y)

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) # 数组扁平化

plt.figure() # 画图

plt.xticks(tuple([x for x in range(7)]))
plt.yticks(tuple([y for y in range(7)if y != 0]))

plt.pcolormesh(xx, yy, Z, cmap=cmap_light) # 不同区域颜色

plt.xlabel('$x^{(1)}$')
plt.ylabel('$x^{(2)}$') # 坐标标签

plt.scatter(x[:, 0], x[:, 1], c=y)

plt.show()