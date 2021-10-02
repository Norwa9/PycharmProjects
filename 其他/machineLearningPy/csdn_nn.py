import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# 导入数据
path = r'/Users/luowei/Documents/机器学习吴恩达/ml答案/machine-learning-ex4/ex4/ex4data1'
data = loadmat(path)

X = data['X']                           # 5000 x 400
y = data['y']                           # 5000 x 1
X = np.mat(X)
y = np.mat(y)

l1 = 400       # 第一层有400个激活单元
l2 = 25        # 第二层有25个激活单元
l3 = 10        # 第三层有10个输出单元

lr = 1

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

# 定义Sigmoid函数
def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义Sigmoid函数的导数
def sigmoid_gradient(z):
    return np.multiply(Sigmoid(z), (1 - Sigmoid(z)))

# 随机初始化
def random(size):
    return np.random.uniform(-0.12, 0.12, size)

# 定义前向传播
def ForwardPropagation(X, theta1, theta2):

    a1 = np.insert(X, 0, 1, axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(Sigmoid(z2), 0, 1, axis=1)
    z3 = a2 * theta2.T
    h = Sigmoid(z3)

    return a1, z2, a2, z3, h

params = random(l2 * (l1 + 1) + l3 * (l2 + 1))      # 将参赛展开并进行初始化，大小为(25*401 + 10*26)

# 定义反向传播
def BackPropagation(params, l1, l2, l3, X, y, lr):

    theta1 = np.mat(np.reshape(params[:l2 * (l1 + 1)], (l2, (l1 + 1))))     # 参数还原
    theta2 = np.mat(np.reshape(params[l2 * (l1 + 1):], (l3, (l2 + 1))))

    a1, z2, a2, z3, h = ForwardPropagation(X, theta1, theta2)     # 利用前向传播得到输出结果h

    delta1 = np.zeros(theta1.shape)  # (25, 401)                  # 初始化误差矩阵为0
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # 代价函数
    J = 0
    for i in range(5000):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / 5000

    #正则化
    J += (float(lr) / (2 * 5000)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    for t in range(5000):                 # 对每个样本依次进行操作
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)         # 得到输出层的误差

        z2t = np.insert(z2t, 0, 1, axis=1)  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)    # 得到隐藏层的误差

        delta1 = delta1 + (d2t[:, 1:]).T * a1t      # 得到误差矩阵
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / 5000               # 对误差矩阵求均值
    delta2 = delta2 / 5000

    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * lr) / 5000   # 加上正则化，并去掉偏置项
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * lr) / 5000

    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    print(np.shape(grad))
    return J, grad

from scipy.optimize import minimize

fmin = minimize(fun=BackPropagation, x0=params, args=(l1, l2, l3, X, y_onehot, lr),
                method='TNC', jac=True, options={'maxiter': 250})

theta1 = np.mat(np.reshape(fmin.x[:l2 * (l1 + 1)], (l2, (l1 + 1))))      # 优化好的参数
theta2 = np.mat(np.reshape(fmin.x[l2 * (l1 + 1):], (l3, (l2 + 1))))

def pred_accuracy():
    a1, z2, a2, z3, h = ForwardPropagation(X, theta1, theta2)
    y_pred = np.array(np.argmax(h, axis=1) + 1)

    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print ('accuracy = {0}%'.format(accuracy * 100))

# 可视化隐藏层
def plot_hidden(theta1):
    t1 = theta1[:, 1:]
    fig, ax = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6, 6))
    for r in range(5):
        for c in range(5):
            ax[r, c].matshow(t1[r * 5 + c].reshape(20, 20), cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
    plt.show()

