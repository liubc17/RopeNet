import torch
import torch.nn as nn
import torch.nn.functional as F
from rpca_layer import rpca_conv, rpca_fc


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, bias=False)
        # print(self.conv1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, 5, bias=False)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # with dropout
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))  # without dropout
        x = self.bn2(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)  # with dropout
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


class rpca_LeNet(nn.Module):
    """rpca_LeNet with rpca_conv and rpca_fc replacing convolution and fully connected layer of LeNet"""
    def __init__(self, load_dir):
        super(rpca_LeNet, self).__init__()
        self.conv1 = rpca_conv(1, 10, 5, load_dir=load_dir, layer_name='conv1', padding=0)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = rpca_conv(10, 20, 5, load_dir=load_dir, layer_name='conv2', padding=0)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = rpca_fc(320, 50, load_dir=load_dir, layer_name='fc1')
        self.fc2 = rpca_fc(50, 10, load_dir=load_dir, layer_name='fc2')

    def forward(self, x, decompose=False, sparsity=1):
        x = F.relu(F.max_pool2d(self.conv1(x, decompose=False, sparsity=sparsity), 2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x, decompose=False, sparsity=sparsity)), 2))  # with dropout
        # x = F.relu(F.max_pool2d(self.conv2(x, decompose=decompose, sparsity=sparsity), 2))  # without dropout
        x = self.bn2(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x, decompose=decompose, sparsity=sparsity))
        x = F.dropout(x, training=self.training)  # with dropout
        x = F.relu(self.fc2(x, decompose=decompose, sparsity=sparsity))
        return F.log_softmax(x, dim=1)

