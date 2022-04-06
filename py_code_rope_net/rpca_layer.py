import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as io


class rpca_conv(nn.Module):
    """ rpca_conv is a convolution layer with two branches, one for ordinary convolution, another for
        rpca convolution. US and V for SVD decomposition of rpca, E for sparse term of rpca, E_5 ~ E_9
        for different sparsity of E. The sparsity of the sparse term is assigned during reference.
        During forward, if decompose=False, the layer executes ordinary convolution. If decompose=True,
        the layer executes rpca convolution, which consists of a low rank convolution and a sparse convolution.
        Args:
            in_ch: input channel of the layer
            out_ch: output channel of the layer
            kernel_size: convolution kernel size of the layer
            layer_name: layer name for the weight_mnist of the import layer after rpca decomposition
    """
    def __init__(self, in_ch, out_ch, kernel_size, load_dir, layer_name, stride=1, padding=1, bias=False):
        super(rpca_conv, self).__init__()
        weights = io.loadmat(load_dir + layer_name + '.mat')
        US = torch.from_numpy(weights['US'])
        US = US.view(out_ch, -1, 1, 1)
        V = torch.from_numpy(weights['V'])
        V = V.view(-1, in_ch, kernel_size, kernel_size)

        E = torch.from_numpy(weights['E'])
        E = E.view(out_ch, in_ch, kernel_size, kernel_size)
        position = torch.nonzero(E)
        mask = E.clone().detach()
        mask[position.numpy().transpose(1, 0)] = 1

        E_5 = torch.from_numpy(weights['E_5'])
        E_5 = E_5.view(out_ch, in_ch, kernel_size, kernel_size)
        position5 = torch.nonzero(E_5)
        mask5 = E_5.clone().detach()
        mask5[position5.numpy().transpose(1, 0)] = 1

        E_6 = torch.from_numpy(weights['E_6'])
        E_6 = E_6.view(out_ch, in_ch, kernel_size, kernel_size)
        position6 = torch.nonzero(E_6)
        mask6 = E_6.clone().detach()
        mask6[position6.numpy().transpose(1, 0)] = 1

        E_7 = torch.from_numpy(weights['E_7'])
        E_7 = E_7.view(out_ch, in_ch, kernel_size, kernel_size)
        position7 = torch.nonzero(E_7)
        mask7 = E_7.clone().detach()
        mask7[position7.numpy().transpose(1, 0)] = 1

        E_8 = torch.from_numpy(weights['E_8'])
        E_8 = E_8.view(out_ch, in_ch, kernel_size, kernel_size)
        position8 = torch.nonzero(E_8)
        mask8 = E_8.clone().detach()
        mask8[position8.numpy().transpose(1, 0)] = 1

        E_9 = torch.from_numpy(weights['E_9'])
        E_9 = E_9.view(out_ch, in_ch, kernel_size, kernel_size)
        position9 = torch.nonzero(E_9)
        mask9 = E_9.clone().detach()
        mask9[position9.numpy().transpose(1, 0)] = 1

        E_95 = torch.from_numpy(weights['E_95'])
        E_95 = E_95.view(out_ch, in_ch, kernel_size, kernel_size)
        position95 = torch.nonzero(E_95)
        mask95 = E_95.clone().detach()
        mask95[position95.numpy().transpose(1, 0)] = 1

        E_98 = torch.from_numpy(weights['E_98'])
        E_98 = E_98.view(out_ch, in_ch, kernel_size, kernel_size)
        position98 = torch.nonzero(E_98)
        mask98 = E_98.clone().detach()
        mask98[position98.numpy().transpose(1, 0)] = 1

        E_99 = torch.from_numpy(weights['E_99'])
        E_99 = E_99.view(out_ch, in_ch, kernel_size, kernel_size)
        position99 = torch.nonzero(E_99)
        mask99 = E_99.clone().detach()
        mask99[position99.numpy().transpose(1, 0)] = 1

        if bias:
            b = torch.from_numpy(weights['bias'])
            b = b.view(out_ch)
            self.b = nn.Parameter(b)
        assert US.shape[0] == out_ch
        assert US.shape[1] == V.shape[0]

        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=self.stride, padding=padding, bias=bias)
        self.weight_US = nn.Parameter(US)
        self.weight_V = nn.Parameter(V)
        self.weight_E = nn.Parameter(E)
        self.weight_E_5 = nn.Parameter(E_5)
        self.weight_E_6 = nn.Parameter(E_6)
        self.weight_E_7 = nn.Parameter(E_7)
        self.weight_E_8 = nn.Parameter(E_8)
        self.weight_E_9 = nn.Parameter(E_9)
        self.weight_E_95 = nn.Parameter(E_95)
        self.weight_E_98 = nn.Parameter(E_98)
        self.weight_E_99 = nn.Parameter(E_99)
        self.mask = mask.cuda()
        self.mask5 = mask5.cuda()
        self.mask6 = mask6.cuda()
        self.mask7 = mask7.cuda()
        self.mask8 = mask8.cuda()
        self.mask9 = mask9.cuda()
        self.mask95 = mask95.cuda()
        self.mask98 = mask98.cuda()
        self.mask99 = mask99.cuda()

        # a test for separated convolution layer
        # self.weight_USV = nn.Parameter(USV)
        # self.W = nn.Parameter(USV + E)
        # W = torch.mm(torch.from_numpy(weight_mnist['US']), torch.from_numpy(weight_mnist['V'])) + torch.from_numpy(weight_mnist['E'])
        # W = W.view(out_ch, in_ch, kernel_size, kernel_size)
        # print(W)
        # self.bias = nn.Parameter(b)

    def forward(self, x, decompose=True, sparsity=1):
        # a test for separated convolution layer
        # if decompose:
        #     out = F.conv2d(x, self.W, self.bias, stride=1, padding=0)
        #     return out
        if decompose:
            vx = F.conv2d(x, self.weight_V, stride=self.stride, padding=self.padding)
            if self.bias:
                Lx = F.conv2d(vx, self.weight_US, self.b, stride=1, padding=0)
            else:
                Lx = F.conv2d(vx, self.weight_US, stride=1, padding=0)
            if sparsity == 1:
                Ex = F.conv2d(x, torch.mul(self.weight_E, self.mask), stride=self.stride, padding=self.padding)

            elif sparsity == 0.5:
                Ex = F.conv2d(x, torch.mul(self.weight_E_5, self.mask5), stride=self.stride, padding=self.padding)

            elif sparsity == 0.6:
                Ex = F.conv2d(x, torch.mul(self.weight_E_6, self.mask6), stride=self.stride, padding=self.padding)

            elif sparsity == 0.7:
                Ex = F.conv2d(x, torch.mul(self.weight_E_7, self.mask7), stride=self.stride, padding=self.padding)

            elif sparsity == 0.8:
                Ex = F.conv2d(x, torch.mul(self.weight_E_8, self.mask8), stride=self.stride, padding=self.padding)

            elif sparsity == 0.9:
                Ex = F.conv2d(x, torch.mul(self.weight_E_9, self.mask9), stride=self.stride, padding=self.padding)

            elif sparsity == 0.95:
                Ex = F.conv2d(x, torch.mul(self.weight_E_95, self.mask95), stride=self.stride, padding=self.padding)

            elif sparsity == 0.98:
                Ex = F.conv2d(x, torch.mul(self.weight_E_98, self.mask98), stride=self.stride, padding=self.padding)

            elif sparsity == 0.99:
                Ex = F.conv2d(x, torch.mul(self.weight_E_99, self.mask99), stride=self.stride, padding=self.padding)

            out = Lx + Ex
            return out

        else:
            out = self.conv(x)
            return out


class rpca_fc(nn.Module):
    """ rpca_fc is a fully connected layer with two branches, one for ordinary fully connection, another
        for rpca fully connection. US and V for SVD decomposition of rpca, E for sparse term of rpca,
        E_5 ~ E_9 for different sparseness of E. The sparsity of the sparse term is assigned during reference.
        During forward, if decompose=False, the layer executes ordinary fully connection. If decompose=True,
        the layer executes rpca fully connection, which consists of a low rank fully connection and a sparse
        fully connection.
        Args:
            in_feature: input dimension of the layer
            out_feature: output dimension of the layer
            layer_name: layer name for the weight_mnist of the import layer after rpca decomposition
    """
    def __init__(self, in_feature, out_feature, load_dir, layer_name):
        super(rpca_fc, self).__init__()
        weights = io.loadmat(load_dir + layer_name + '.mat')
        US = torch.from_numpy(weights['US'])
        US = US.transpose(0, 1)
        V = torch.from_numpy(weights['V'])
        V = V.transpose(0, 1)

        E = torch.from_numpy(weights['E'])
        E = E.transpose(0, 1)
        position = torch.nonzero(E)
        mask = E.clone().detach()
        mask[position.numpy().transpose(1, 0)] = 1

        E_5 = torch.from_numpy(weights['E_5'])
        E_5 = E_5.transpose(0, 1)
        position5 = torch.nonzero(E_5)
        mask5 = E_5.clone().detach()
        mask5[position5.numpy().transpose(1, 0)] = 1

        E_6 = torch.from_numpy(weights['E_6'])
        E_6 = E_6.transpose(0, 1)
        position6 = torch.nonzero(E_6)
        mask6 = E_6.clone().detach()
        mask6[position6.numpy().transpose(1, 0)] = 1

        E_7 = torch.from_numpy(weights['E_7'])
        E_7 = E_7.transpose(0, 1)
        position7 = torch.nonzero(E_7)
        mask7 = E_7.clone().detach()
        mask7[position7.numpy().transpose(1, 0)] = 1

        E_8 = torch.from_numpy(weights['E_8'])
        E_8 = E_8.transpose(0, 1)
        position8 = torch.nonzero(E_8)
        mask8 = E_8.clone().detach()
        mask8[position8.numpy().transpose(1, 0)] = 1

        E_9 = torch.from_numpy(weights['E_9'])
        E_9 = E_9.transpose(0, 1)
        position9 = torch.nonzero(E_9)
        mask9 = E_9.clone().detach()
        mask9[position9.numpy().transpose(1, 0)] = 1

        E_95 = torch.from_numpy(weights['E_95'])
        E_95 = E_95.transpose(0, 1)
        position95 = torch.nonzero(E_95)
        mask95 = E_95.clone().detach()
        mask95[position95.numpy().transpose(1, 0)] = 1

        E_98 = torch.from_numpy(weights['E_98'])
        E_98 = E_98.transpose(0, 1)
        position98 = torch.nonzero(E_98)
        mask98 = E_98.clone().detach()
        mask98[position98.numpy().transpose(1, 0)] = 1

        E_99 = torch.from_numpy(weights['E_99'])
        E_99 = E_99.transpose(0, 1)
        position99 = torch.nonzero(E_99)
        mask99 = E_99.clone().detach()
        mask99[position99.numpy().transpose(1, 0)] = 1

        b = torch.from_numpy(weights['bias'])

        self.fc = nn.Linear(in_feature, out_feature)
        self.weight_US = nn.Parameter(US)
        self.weight_V = nn.Parameter(V)
        self.weight_E = nn.Parameter(E)
        self.weight_E_5 = nn.Parameter(E_5)
        self.weight_E_6 = nn.Parameter(E_6)
        self.weight_E_7 = nn.Parameter(E_7)
        self.weight_E_8 = nn.Parameter(E_8)
        self.weight_E_9 = nn.Parameter(E_9)
        self.weight_E_95 = nn.Parameter(E_95)
        self.weight_E_98 = nn.Parameter(E_98)
        self.weight_E_99 = nn.Parameter(E_99)
        self.mask = mask.cuda()
        self.mask5 = mask5.cuda()
        self.mask6 = mask6.cuda()
        self.mask7 = mask7.cuda()
        self.mask8 = mask8.cuda()
        self.mask9 = mask9.cuda()
        self.mask95 = mask95.cuda()
        self.mask98 = mask98.cuda()
        self.mask99 = mask99.cuda()
        self.bias = nn.Parameter(b)

    def forward(self, x, decompose=True, sparsity=1):
        if decompose:
            vx = torch.mm(x, self.weight_V)
            Lx = torch.mm(vx, self.weight_US) + self.bias.repeat(x.shape[0], 1)
            if sparsity == 1:
                Ex = torch.mm(x, torch.mul(self.weight_E, self.mask))

            elif sparsity == 0.5:
                Ex = torch.mm(x, torch.mul(self.weight_E_5, self.mask5))

            elif sparsity == 0.6:
                Ex = torch.mm(x, torch.mul(self.weight_E_6, self.mask6))

            elif sparsity == 0.7:
                Ex = torch.mm(x, torch.mul(self.weight_E_7, self.mask7))

            elif sparsity == 0.8:
                Ex = torch.mm(x, torch.mul(self.weight_E_8, self.mask8))

            elif sparsity == 0.9:
                Ex = torch.mm(x, torch.mul(self.weight_E_9, self.mask9))

            elif sparsity == 0.95:
                Ex = torch.mm(x, torch.mul(self.weight_E_95, self.mask95))

            elif sparsity == 0.98:
                Ex = torch.mm(x, torch.mul(self.weight_E_98, self.mask98))

            elif sparsity == 0.99:
                Ex = torch.mm(x, torch.mul(self.weight_E_99, self.mask99))

            out = Lx + Ex
            return out

        else:
            out = self.fc(x)
            return out
