import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Dsigmoid(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

# 随机初始化
def random(size):
    return np.random.uniform(-0.12, 0.12, size)

path = r'/Users/luowei/Documents/机器学习吴恩达/ml答案/machine-learning-ex4/ex4/ex4data1'
data = loadmat(path)

X = data['X']  # 5000 x 400
y = data['y']  # 5000 x 1
X = np.mat(X)
y = np.mat(y)
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)  # 转换独热码 5000 x 10

l1 = 400
l2 = 25
l3 = 10
lamda = 1
params = random(l2 * (l1 + 1) + l3 * (l2 + 1))  # 将参赛展开并进行初始化，大小为(25*401 + 10*26)

# path = r'C:\Users\PC\Desktop\Norwa9\data\Andrew-NG-Meachine-Learning\machine-learning-ex4\ex4\ex4weights'
# weights = loadmat(path)
# theta1 = weights['Theta1']
# theta2 = weights['Theta2']

# 前向传播
def forwardPropagation(X, theta1, theta2):
    a1 = np.insert(X, 0, 1, axis=1)  # 添加偏置项1
    # print('X',X)
    # print('a1',a1)
    z2 = np.dot(a1, theta1.T)
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)  # 添加偏置项1
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

# 输出层代价函数
def costFuncation(y,h,theta1,theta2):
    J = 0
    # 向量化写法
    first = -np.multiply(y, np.log(h))  # (5000 x 10) .* (5000 x 10) = (5000 x 10)
    second = -np.multiply((1 - y), np.log(1 - h))  # (5000 x 10) .* (5000 x 10) = (5000 x 10)
    J = np.sum(first + second)

    J = J / 5000

    # 添加正则化项
    J += (float(lamda) / (2 * 5000)) * ( np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)) )
    print('J:',J)
    return J

# 反向传播
def backwardPropagation(params, l1, l2, l3, X, y, lamda):
    theta1 = np.mat(np.reshape(params[:l2 * (l1 + 1)], (l2, (l1 + 1))))  # theta1: 25 x 401
    theta2 = np.mat(np.reshape(params[l2 * (l1 + 1):], (l3, (l2 + 1))))  # theta2: 10 x 26

    a1, z2, a2, z3, h = forwardPropagation(X, theta1, theta2)
    # a1 : 5000 * 401
    # a2 : 5000 x 26

    # 初始化误差矩阵为0
    delta1 = np.zeros(theta1.shape)  # 25 x 401
    delta2 = np.zeros(theta2.shape)  # 10 x 26

    J = costFuncation(y, h, theta1, theta2)

    d3 = h - y  # 5000 x 10
    z2 = np.insert(z2, 0, 1, axis = 1) # z2: 5000 x 26
    d2 = np.multiply(d3 * theta2, Dsigmoid(z2)) # d2: 5000 x 26 # 隐藏层误差

    delta1 = (a1.T * d2[:,1:]).T / 5000 # 得到误差矩阵，并去掉偏置项:25 x 401
    delta2 = (a2.T * d3).T / 5000 # 10 x 26

    delta1[:,1:] = delta1[:,1:] + (theta1[:, 1:] * lamda) / 5000
    delta2[:,1:] = delta2[:,1:] + (theta2[:, 1:] * lamda) / 5000

    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2))) # 沿现有轴合并数组序列。
    # print('np.shape(grad):',np.shape(grad))
    return J,grad

fmin = minimize(
    fun=backwardPropagation,
    x0=params,
    args=(l1,l2,l3,X,y_onehot,lamda),
    method='TNC',
    jac=True,
    options={'maxiter': 250}
)

# 优化好的参数：
theta1 = np.mat(np.reshape(fmin.x[:l2 * (l1 + 1)], (l2, (l1+1))))
theta2 = np.mat(np.reshape(fmin.x[l2 * (l1 + 1):], (l3, (l2+1))))

def pred_accuracy():
    a1, z2, a2, z3, h = forwardPropagation(X,theta1,theta2)
    y_pred = np.array((np.argmax(h, axis=1) + 1)) # argmax返回某个轴上最大值的下标，示例见文档.因为下标从0开始，所以+1
    correct = [1 if a==b else 0 for a,b in zip(y_pred, y)] # zip:分别从y_pred和y中取一个元素组成一个元组。
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('准确率 = {0}%'.format(accuracy * 100))

# 可视化数据
def DisplayData(X):
    fig, ax = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True)  # 生成10行10列的画布
    pick_one = np.random.randint(0, 5000, (100,))  # 随机生成100个0到5000的数字，作为索引
    # print('pick_one:', pick_one)
    for row in range(10):
        for col in range(10):
            # 最后加上转置，使得混乱的数字变得整齐
            x = (X[pick_one[col + row * 10]].reshape((20, 20))).T  # 提取对应的数字信息
            ax[row, col].matshow(x, cmap='gray_r')  # 画出灰度图

    plt.xticks([])
    plt.yticks([])
    plt.show()

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

# DisplayData(X)
pred_accuracy()
# plot_hidden(theta1)

