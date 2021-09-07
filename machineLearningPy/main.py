import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

path = r'C:\Users\PC\Desktop\Norwa9\data\Andrew-NG-Meachine-Learning\machine-learning-ex4\ex4\ex4data1'
data = loadmat(path)

X = data['X']  # 5000 x 400
y = data['y']  # 5000 x 1

path = r'C:\Users\PC\Desktop\Norwa9\data\Andrew-NG-Meachine-Learning\machine-learning-ex4\ex4\ex4weights'
weights = loadmat(path)
theta1 = weights['Theta1']
theta2 = weights['Theta2']

# 转换独热码
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)  # 5000 x 10


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 前向传播
def forwardPropagation(X, theta1, theta2):
    a1 = np.insert(X, 0, 1, axis=1)  # 添加偏置项1
    # print('X',X)
    # print('a1',a1)
    z2 = np.dot(a1, theta1.T)
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)  # 添加偏置项1
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return h


# 逻辑回归代价函数
def costFuncation(X, y, lamda):
    X = np.mat(X)
    y = np.mat(y)
    h = forwardPropagation(X, theta1, theta2)  # 5000 x 10

    J = 0
    # 向量化写法
    first = -np.multiply(y, np.log(h))  # (5000 x 10) .* (5000 x 10) = (5000 x 10)
    second = -np.multiply((1 - y), np.log(1 - h))  # (5000 x 10) .* (5000 x 10) = (5000 x 10)
    # print('first:\n',first)
    # print('second:\n', second)
    J = np.sum(first + second)

    J = J / 5000
    # 添加正则化项
    J += (float(lamda) / 5000) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    return J

# 反向传播
def backwardPropagation()

# 可视化数据
def DisplayData(X):
    fig, ax = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True)  # 生成10行10列的画布
    pick_one = np.random.randint(0, 5000, (100,))  # 随机生成100个0到5000的数字，作为索引
    print('pick_one:', pick_one)
    for row in range(10):
        for col in range(10):
            # 最后加上转置，使得混乱的数字变得整齐
            x = (X[pick_one[col + row * 10]].reshape((20, 20))).T  # 提取对应的数字信息
            ax[row, col].matshow(x, cmap='gray_r')  # 画出灰度图

    plt.xticks([])
    plt.yticks([])
    plt.show()


# DisplayData(X)

lamda = 1
print(costFuncation(X, y, lamda))
