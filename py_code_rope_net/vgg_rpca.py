'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from rpca_layer import rpca_conv, rpca_fc

__all__ = ['rpca_vgg11', 'rpca_vgg13', 'rpca_vgg16', 'rpca_vgg19']


class rpca_VGG(nn.Module):
    '''
    rpca_VGG model
    '''
    def __init__(self, load_dir, depth, num_class=10):
        super(rpca_VGG, self).__init__()
        self.conv1_1 = rpca_conv(3, 64, kernel_size=3, load_dir=load_dir, layer_name='conv1_1', bias=True)
        if depth > 11:
            self.conv1_2 = rpca_conv(64, 64, kernel_size=3, load_dir=load_dir, layer_name='conv1_2', bias=True)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = rpca_conv(64, 128, kernel_size=3, load_dir=load_dir, layer_name='conv2_1', bias=True)
        if depth > 11:
            self.conv2_2 = rpca_conv(128, 128, kernel_size=3, load_dir=load_dir, layer_name='conv2_2', bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = rpca_conv(128, 256, kernel_size=3, load_dir=load_dir, layer_name='conv3_1', bias=True)
        self.conv3_2 = rpca_conv(256, 256, kernel_size=3, load_dir=load_dir, layer_name='conv3_2', bias=True)
        if depth > 13:
            self.conv3_3 = rpca_conv(256, 256, kernel_size=3, load_dir=load_dir, layer_name='conv3_3', bias=True)
        if depth > 16:
            self.conv3_4 = rpca_conv(256, 256, kernel_size=3, load_dir=load_dir, layer_name='conv3_4', bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = rpca_conv(256, 512, kernel_size=3, load_dir=load_dir, layer_name='conv4_1', bias=True)
        self.conv4_2 = rpca_conv(512, 512, kernel_size=3, load_dir=load_dir, layer_name='conv4_2', bias=True)
        if depth > 13:
            self.conv4_3 = rpca_conv(512, 512, kernel_size=3, load_dir=load_dir, layer_name='conv4_3', bias=True)
        if depth > 16:
            self.conv4_4 = rpca_conv(512, 512, kernel_size=3, load_dir=load_dir, layer_name='conv4_4', bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = rpca_conv(512, 512, kernel_size=3, load_dir=load_dir, layer_name='conv5_1', bias=True)
        self.conv5_2 = rpca_conv(512, 512, kernel_size=3, load_dir=load_dir, layer_name='conv5_2', bias=True)
        if depth > 13:
            self.conv5_3 = rpca_conv(512, 512, kernel_size=3, load_dir=load_dir, layer_name='conv5_3', bias=True)
        if depth > 16:
            self.conv5_4 = rpca_conv(512, 512, kernel_size=3, load_dir=load_dir, layer_name='conv5_4', bias=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = rpca_fc(512, 512, load_dir=load_dir, layer_name='fc1')
        self.fc2 = rpca_fc(512, 512, load_dir=load_dir, layer_name='fc2')
        self.fc3 = rpca_fc(512, num_class, load_dir=load_dir, layer_name='fc3')
        self.depth = depth

    def forward(self, x, decompose=True, sparsity=1):
        out = F.relu(self.conv1_1(x, decompose=decompose, sparsity=sparsity), inplace=True)
        if self.depth > 11:
            out = F.relu(self.conv1_2(out, decompose=decompose, sparsity=sparsity), inplace=True)
        out = self.pool0(out)
        out = F.relu(self.conv2_1(out, decompose=decompose, sparsity=sparsity), inplace=True)
        if self.depth > 11:
            out = F.relu(self.conv2_2(out, decompose=decompose, sparsity=sparsity), inplace=True)
        out = self.pool1(out)
        out = F.relu(self.conv3_1(out, decompose=decompose, sparsity=sparsity), inplace=True)
        out = F.relu(self.conv3_2(out, decompose=decompose, sparsity=sparsity), inplace=True)
        if self.depth > 13:
            out = F.relu(self.conv3_3(out, decompose=decompose, sparsity=sparsity), inplace=True)
        if self.depth > 16:
            out = F.relu(self.conv3_4(out, decompose=decompose, sparsity=sparsity), inplace=True)
        out = self.pool2(out)
        out = F.relu(self.conv4_1(out, decompose=decompose, sparsity=sparsity), inplace=True)
        out = F.relu(self.conv4_2(out, decompose=decompose, sparsity=sparsity), inplace=True)
        if self.depth > 13:
            out = F.relu(self.conv4_3(out, decompose=decompose, sparsity=sparsity), inplace=True)
        if self.depth > 16:
            out = F.relu(self.conv4_4(out, decompose=decompose, sparsity=sparsity), inplace=True)
        out = self.pool3(out)
        out = F.relu(self.conv5_1(out, decompose=decompose, sparsity=sparsity), inplace=True)
        out = F.relu(self.conv5_2(out, decompose=decompose, sparsity=sparsity), inplace=True)
        if self.depth > 13:
            out = F.relu(self.conv5_3(out, decompose=decompose, sparsity=sparsity), inplace=True)
        if self.depth > 16:
            out = F.relu(self.conv5_4(out, decompose=decompose, sparsity=sparsity), inplace=True)
        out = self.pool4(out)
        out = out.view(out.size(0), -1)
        out = F.dropout(out)
        out = F.dropout(F.relu(self.fc1(out, decompose=decompose, sparsity=sparsity)))
        out = F.relu(self.fc2(out, decompose=decompose, sparsity=sparsity))
        out = self.fc3(out, decompose=decompose, sparsity=sparsity)
        return out


def rpca_vgg11(load_dir, num_class=10):
    """rpca_VGG 11"""
    return rpca_VGG(load_dir, 11, num_class=num_class)


# def vgg11_bn():
#     """VGG 11-layer model (configuration "A") with batch normalization"""
#     return VGG(make_layers(cfg['A'], batch_norm=True))


def rpca_vgg13(load_dir, num_class=10):
    """rpca_VGG 13"""
    return rpca_VGG(load_dir, 13, num_class=num_class)


# def vgg13_bn():
#     """VGG 13-layer model (configuration "B") with batch normalization"""
#     return VGG(make_layers(cfg['B'], batch_norm=True))

def rpca_vgg16(load_dir, num_class=10):
    """rpca_VGG 16"""
    return rpca_VGG(load_dir, 16, num_class=num_class)


# def vgg16_bn():
#     """VGG 16-layer model (configuration "D") with batch normalization"""
#     return VGG(make_layers(cfg['D'], batch_norm=True))


def rpca_vgg19(load_dir, num_class=10):
    """rpca_VGG 19"""
    return rpca_VGG(load_dir, 19, num_class=num_class)


# def vgg19_bn():
#     """VGG 19-layer model (configuration 'E') with batch normalization"""
#     return VGG(make_layers(cfg['E'], batch_norm=True))
