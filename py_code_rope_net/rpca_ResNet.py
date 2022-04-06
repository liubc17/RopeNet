'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from rpca_layer import rpca_conv, rpca_fc
from torch.autograd import Variable


__all__ = ['rpca_ResNet', 'rpca_resnet20', 'rpca_resnet32', 'rpca_resnet44', 'rpca_resnet56', 'rpca_resnet110', 'rpca_resnet1202']


def _weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class rpca_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, index1, index2, load_dir, stride=1, option='A'):
        super(rpca_BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = rpca_conv(in_planes, planes, kernel_size=3, load_dir=load_dir,
                               layer_name=str(index1) + '_' + str(index2) + '_conv1',
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = rpca_conv(planes, planes, kernel_size=3, load_dir=load_dir,
                               layer_name=str(index1) + '_' + str(index2) + '_conv2',
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, input):
        x, decompose, sparsity = input[0], input[1], input[2]
        out = F.relu(self.bn1(self.conv1(x, decompose=decompose, sparsity=sparsity)))
        out = self.bn2(self.conv2(out, decompose=decompose, sparsity=sparsity))
        out += self.shortcut(x)
        out = F.relu(out)
        return [out, decompose, sparsity]


class rpca_ResNet(nn.Module):
    def __init__(self, block, num_blocks, load_dir, num_classes=10):
        super(rpca_ResNet, self).__init__()
        self.in_planes = 16
        self.load_dir = load_dir

        self.conv1 = rpca_conv(3, 16, kernel_size=3, load_dir=load_dir, layer_name='conv1', stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, index1=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, index1=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, index1=3)
        # self.linear = nn.Linear(64, num_classes)
        self.linear = rpca_fc(64, num_classes, load_dir=load_dir, layer_name='linear')
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, index1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, index1=index1, index2=i, load_dir=self.load_dir, stride=stride))
            self.in_planes = planes * block.expansion
            i += 1

        return nn.Sequential(*layers)

    def forward(self, x, decompose=True, sparsity=1):
        out = F.relu(self.bn1(self.conv1(x, decompose=decompose, sparsity=sparsity)))
        input1 = [out, decompose, sparsity]
        out = self.layer1(input1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out[0]
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out, decompose=decompose, sparsity=sparsity)
        return out


def rpca_resnet20(load_dir, num_classes=10):
    return rpca_ResNet(rpca_BasicBlock, [3, 3, 3], load_dir, num_classes=num_classes)


def rpca_resnet32(load_dir, num_classes=10):
    return rpca_ResNet(rpca_BasicBlock, [5, 5, 5], load_dir, num_classes=num_classes)


def rpca_resnet44(load_dir, num_classes=10):
    return rpca_ResNet(rpca_BasicBlock, [7, 7, 7], load_dir, num_classes=num_classes)


def rpca_resnet56(load_dir, num_classes=10):
    return rpca_ResNet(rpca_BasicBlock, [9, 9, 9], load_dir, num_classes=num_classes)


def rpca_resnet110(load_dir, num_classes=10):
    return rpca_ResNet(rpca_BasicBlock, [18, 18, 18], load_dir, num_classes=num_classes)


def rpca_resnet1202(load_dir, num_classes=10):
    return rpca_ResNet(rpca_BasicBlock, [200, 200, 200], load_dir, num_classes=num_classes)
