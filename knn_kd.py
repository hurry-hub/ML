# k近邻算法
# 算法过程：1.根据距离度量，在训练集找到与x最邻近的k个点，涵盖这k个点的邻域为Nk(x)
#          2.在Nk(x)中根据分类决策规则（多数表决）决定x的类别y：y=argmax(sigama)I(yi=cj)
# k近邻算法没有学习过程
# 因为太慢了，只测了100个点
# k=25,accuracy=98%,time=85.1s


import numpy as np
import time
import operator


# 加载数据过程类似
def loadData(filename):
    print('start to read data')

    image_array = []
    label_array = []

    file = open(filename, 'r')

    for line in file.readlines():
        # 数据格式：每个样本一行，以','为间隔，以'\n'结尾的字符串,首字符为类别，后面跟28*28个像素值
        curline = line.strip().split(',')
        # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        label_array.append(int(curline[0]))
        image = [int(num) for num in curline[1:]]
        # 这里并没有用除255，只在意它的相对大小，整数计算比浮点数快。
        image_array.append(image)
    return image_array, label_array


# 距离度量
def eucl_dist(x1, x2):
    result = np.sqrt(np.sum(np.square(x1 - x2)))
    # print('result',result)
    return result


# 线性扫描，实在是太慢了
'''
采用线性扫描的方式，因为构造kd树太复杂
def get_closest(image_array,label_array,x,k):

    distance_list=[0]*len(image_array)
    for i in range(len(image_array)):
        image=image_array[i]
        distance=eucl_dis(image,x)
        distance_list[i]=distance
    sort_index=np.argsort(np.array(distance_list))
    top_k_index=sort_index[:k]
    label_list=[0]*10
    for index in top_k_index:
        label_list[label_array[index]]+=1
    return label_list.index(max(label_list))
'''


# kd树

class Node:
    def __init__(self, parent=None, data=None, left=None, right=None, feature=-1, is_left=True):
        self.parent = parent
        self.left = left
        self.right = right
        self.data = data
        self.feature = feature
        self.is_left = is_left
        self.once = False


# 先定义一个节点，若无数据，则该节点data为None，若刚好为一个数据，则把数据放到data里，否则，把数据一分为二，再新建在左右子树新建两个节点，分别在每个节点上处理
def kdtree(head, instance_array, depth, total_feature):  # 60000*(28*28+1)
    instance_num = instance_array.shape[0]
    print("instance_num:", instance_num)
    if instance_num == 0:
        return None
    if instance_num == 1:
        head.data = np.squeeze(instance_array)
        return head

    feature_choice = depth % total_feature
    # print("total_feature:",total_feature)
    feature_col = instance_array[:, feature_choice]
    sort_index = np.argsort(feature_col)
    mid = instance_num // 2
    small_index = sort_index[:mid]
    large_index = sort_index[mid + 1:]
    small_instance = instance_array[small_index]
    large_instance = instance_array[large_index]
    head_instance = instance_array[sort_index[mid]]

    head.data = np.squeeze(head_instance)
    head.feature = feature_choice
    # print("feature_choice:",feature_choice)
    head.left = Node(parent=head, is_left=True)
    head.left = kdtree(head.left, small_instance, depth + 1, total_feature)
    head.right = Node(parent=head, is_left=False)
    head.right = kdtree(head.right, large_instance, depth + 1, total_feature)
    return head


def construct_kdtree(image, label):
    image_array = np.array(image)
    label_array = np.array(label)[:, np.newaxis]
    instance_array = np.hstack((image_array, label_array))
    print("instance_array.shape:", instance_array.shape)
    feature_num = image_array.shape[1]
    head = Node()
    head = kdtree(head, instance_array, 0, feature_num)
    return head


class bpq:
    def __init__(self, length=10, hold_max=False):
        self.data = []
        self.length = length
        self.hold_max = hold_max

    def append(self, point, distance, label):
        self.data.append((point, distance, label))
        self.data.sort(key=operator.itemgetter(1), reverse=self.hold_max)
        self.data = self.data[:self.length]

    def get_data(self):
        return [item[0] for item in self.data]

    def get_label(self):
        labels = [item[2] for item in self.data]
        uniques, counts = np.unique(labels, return_counts=True)
        return uniques[np.argmax(counts)]

    def get_threshold(self):
        # print('length',len(self.data))
        return np.inf if len(self.data) < self.length else self.data[-1][1]

    def full(self):
        return len(self.data) >= self.length


# 先把第一个放进去，再对于进入的node判断是不是比队列里最大的要小，要是是就放进队列

# 先进入叶节点
def knn_search(test_point, node, queue):
    if node:
        node.once = True
        if node.feature != -1:
            if (test_point[node.feature] < node.data[node.feature]):
                if node.left:
                    knn_search(test_point, node.left, queue)
            else:
                if node.right:
                    knn_search(test_point, node.right, queue)
        distance = eucl_dist(test_point, node.data[:-1])
        if distance < queue.get_threshold():
            queue.append(node.data[:-1], distance, node.data[-1])
        if node.parent:
            feature_choice = node.parent.feature
            if np.abs(test_point[feature_choice] - node.parent.data[feature_choice]) < queue.get_threshold():
                if node.is_left and node.parent.right:
                    if not node.parent.right.once:
                        knn_search(test_point, node.parent.right, queue)
                elif not node.is_left and node.parent.left:
                    if not node.parent.left.once:
                        knn_search(test_point, node.parent.left, queue)


def set_once(node):
    if node:
        node.once = False
        set_once(node.left)
        set_once(node.right)


def knn(train_image, train_label, test_image, test_label, k):
    head = construct_kdtree(train_image, train_label)
    instance_num = len(test_image)
    num = 0
    # for i in range(instance_num):
    for i in range(100):
        image = np.array(test_image[i])
        label = np.array(test_label[i])
        # node=knn_to_end(image,head)
        queue = bpq(k)
        set_once(head)
        knn_search(image, head, queue)
        y = queue.get_label()
        print(y, '\n')
        if y == label:
            num += 1
    accuracy = num / instance_num
    return accuracy


if __name__ == "__main__":
    start = time.time()
    train_image_array, train_label_array = loadData('../data/train.csv')
    test_image_array, test_label_array = loadData('../data/test.csv')
    # accuracy=knn_test(train_image_array,train_label_array,test_image_array,test_label_array,k=25)
    accuracy = knn(train_image_array, train_label_array, test_image_array, test_label_array, k=25)
    print('accuracy : %f' % (accuracy * 100), '%')
    end = time.time()
    print('time spend:', end - start)