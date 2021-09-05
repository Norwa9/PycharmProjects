import mnist
import numpy as np
import matplotlib.pyplot as plt

class Conv3x3:
    # A convolution layer using 3x3 filters.

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is a 3d array with dimentions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        # image: matrix of image
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        # 28x28
        self.last_input = input

        # input_im: matrix of image
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        # print(np.shape(output)) #(26, 26, 8)
        return output

    def backprop(self, d_L_d_out, learn_rate):
        # d_L_d_out: the loss gradient for this layer's outputs
        # learn_rate: a float
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # d_L_d_filters[f]: 3x3 matrix
                # d_L_d_out[i, j, f]: num
                # im_region: 3x3 matrix in image
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        # We aren't returning anything here since we use Conv3x3 as
        # the first layer in our CNN. Otherwise, we'd need to return
        # the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
        return None


class MaxPool2:
    # A Max Pooling layer using a pool size of 2.

    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''
        # image: 3d matix of conv layer

        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        # 26x26x8
        self.last_input = input

        # input: 3d matrix of conv layer
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backprop(self, d_L_d_out):
        # d_L_d_out: the loss gradient for the layer's outputs

        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i + i2, j + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input


class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        # input_len: length of input nodes
        # nodes: lenght of ouput nodes

        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        # 3d
        self.last_input_shape = input.shape

        # 3d to 1d
        input = input.flatten()

        # 1d vector after flatting
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases

        # output before softmax
        # 1d vector:(10,)
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        # only 1 element of d_L_d_out is nonzero
        for i, gradient in enumerate(d_L_d_out):
            # k != c, gradient = 0
            # k == c, gradient = 1
            # try to find i when k == c
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            # all gradients are given value with k != c
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            # change the value of k == c
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of out[i] against totals
            # gradients to every weight in every node
            # this is not the final results
            d_t_d_w = self.last_input  # vector
            d_t_d_b = 1
            # 1000 x 10
            d_t_d_inputs = self.weights

            # Gradients of loss against totals
            # d_L_d_t, d_out_d_t, vector, 10 elements
            d_L_d_t = gradient * d_out_d_t

            # Gradients of loss against weights/biases/input
            # (1000, 1) @ (1, 10) to (1000, 10)
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            # (1000, 10) @ (10, 1)
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

            # it will be used in previous pooling layer
            # reshape into that matrix(13x13x8)
            return d_L_d_inputs.reshape(self.last_input_shape)

class ReLu:
    def __init__(self):
        self.Jacobi = None

    def forward(self, input):
        #ReLu
        relu_out = np.zeros(input.shape)
        for map_num in range(input.shape[-1]):
            for r in np.arange(0, input.shape[0]):
                for c in np.arange(0, input.shape[1]):
                    relu_out[r, c, map_num] = np.max(input[r, c, map_num],0)
        self.Jacobi = relu_out.copy()
        self.Jacobi[self.Jacobi > 0] = 1

        return relu_out

    def backprop(self, d_L_d_out):
        # ReLu
        self.Jacobi = self.Jacobi*d_L_d_out
        return self.Jacobi


#卷积层conv->ReLU->池化层pool->全连接层softmax
conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
relu = ReLu()
pool = MaxPool2()  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10



def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    out = conv.forward((image / 255) - 0.5)
    out = relu.forward(out)
    out = pool.forward(out)
    out = softmax.forward(out)



    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

    # out: vertor of probability
    # loss: num
    # acc: 1 or 0


def train(im, label, lr=.005):
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate intial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = relu.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc

#使用当前的模型对输入的测试集进行测试
#输入：测试集，测试集标签
#输出：测试集的平均交叉熵损失，模型预测测试集的准确率
def test(images,labels):
    t_loss = 0
    acc_num = 0
    for (test_img, test_label) in zip(images, labels):
        _, test_l, test_acc = forward(test_img, test_label)
        t_loss += test_l
        acc_num += test_acc

    #当前模型对测试集平均交叉熵损失
    a_test_loss = t_loss / len(images)
    #当前模型对测试集的预测准确率
    accuracy = acc_num / len(labels)
    return a_test_loss,accuracy

def plot_loss_curve(train_l_arr,test_l_arr):
    ax = plt.gca()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    x_values = range(len(train_l_arr))
    ax.plot(x_values, train_l_arr, c="r", label='train loss')  # 训练loss：红色
    ax.plot(x_values, test_l_arr, c="g", label='valid loss')  # 测试loss：蓝色
    plt.legend()
    plt.show()

#训练CNN
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]
train_loss_arr = []
test_loss_arr = []
for epoch in range(3):
    print('--- Epoch %d ---' % (epoch + 1))

    #将测试集和训练集打乱
    permutation_of_train = np.random.permutation(len(train_images))
    train_images = train_images[permutation_of_train]
    train_labels = train_labels[permutation_of_train]

    # Train
    train_loss = 0
    train_num_correct = 0
    # i: index
    # im: image
    # label: label
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            avg_train_loss = train_loss / 100
            print(
                '[Step %d] Past 100 steps: Average Trainning Loss %.6f | Accuracy: %d%%' % (i + 1, avg_train_loss,train_num_correct)
            )
            if i == len(train_images) - 1:
                train_loss_arr.append(avg_train_loss)
            train_loss = 0
            train_num_correct = 0


        #逐个样本进行训练
        l, acc = train(im, label)
        train_loss += l
        train_num_correct += acc

    # 每个epoch结束后，使用当前训练结果对测试集进行预测，计算其平均交叉熵损失
    avg_test_loss, _ = test(test_images, test_labels)
    test_loss_arr.append(avg_test_loss)
    print(
        '[epoch %d]Average Testing Loss %.3f' % (epoch + 1 , avg_test_loss)
    )

plot_loss_curve(train_loss_arr,test_loss_arr)