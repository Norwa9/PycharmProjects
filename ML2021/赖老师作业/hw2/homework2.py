import numpy as np
import matplotlib.pyplot as plt
import mnist

#Data: read MNIST
DATA_PATH = './mnist'
mn = mnist.MNIST(DATA_PATH)
mn.gz = True # Enable loading of gzip-ed files
train_img, train_label = mn.load_training()
test_img, test_label = mn.load_testing()
train_X = np.array(train_img)
train_Y = np.array(train_label).reshape(-1, 1)
test_X = np.array(test_img)
test_Y = np.array(test_label).reshape(-1, 1)

#Hyperparameters
lr = 0.0000001
epochs = 1000

#Model
W = np.zeros((784, 1))

for step in range(epochs):
  pred = train_X.dot(W)
  grad = 2 * (train_Y - pred).T.dot(-train_X).T / len(train_X)
  W = W - lr * grad
  loss = ((train_Y - pred)**2).mean()
  acc = np.equal(test_Y, np.round(test_X.dot(W))).sum() / len(test_X)
  print("[%d/%d]LOSS:%.3f, ACC:%.3f" %(step+1, epochs, loss, acc))
