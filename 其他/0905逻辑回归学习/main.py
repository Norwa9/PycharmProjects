import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)
import testPy


# step 1/5 生成数据
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2) #生成 sample_nums x 2、元素全1的矩阵。
x0 = torch.normal(mean_value * n_data , 1) + bias ##torch.normal()：每个元素都是从 均值=mean_value * n_data，标准差=1的正态分布中随机生成的。
y0 = torch.zeros(sample_nums)
x1 = torch.normal(-sample_nums * n_data , 1) + bias
y1 = torch.ones(sample_nums)
train_x = torch.cat((x0,x1), 0) #torch.cat()：将两张量拼接到一起。0表示按行拼接。
train_y = torch.cat((y0,y1), 0)

# step 2/5 选择模型
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(in_features= 2 , out_features= 1) #nn.Linear()：设置网络中的全连接层。参数分别为输入和输出的二维张量大小
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x