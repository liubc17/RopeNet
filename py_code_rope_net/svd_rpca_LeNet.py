import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.io as io
import pandas as pd
import argparse

from rpca_LeNet import LeNet, rpca_LeNet


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).detach().cpu().numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)


def train(m, out_dir, mode=1, sparsity=1):
    iter_loss = []
    train_losses = []
    test_losses = []
    iter_loss_path = os.path.join(out_dir, "iter_loss.csv")
    epoch_loss_path = os.path.join(out_dir, "epoch_loss.csv")
    if mode == 1:
        nb_epochs = 10
    if mode == 2:
        nb_epochs = 1
    if mode == 3:
        nb_epochs = 10
    last_loss = 99999
    mkdirs(os.path.join(out_dir, "models"))
    optimizer = optim.SGD(m.parameters(), lr=0.003, momentum=0.9)
    best_test_acc = 0.
    for epoch in range(nb_epochs):
        if mode != 2:
            train_loss = 0.
            train_acc = 0.
            m.train(mode=True)
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                if mode == 1:
                    output = m(data)
                if mode == 3:
                    output = m(data, decompose=True, sparsity=sparsity)
                loss = criterion(output, target)
                loss_value = loss.item()
                iter_loss.append(loss_value)
                train_loss += loss_value
                loss.backward()
                optimizer.step()
                acc = compute_accuray(output, target)
                train_acc += acc
            train_losses.append(train_loss / len(train_loader))

        test_loss = 0.
        test_acc = 0.

        m.train(mode=False)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if mode == 1:
                output = m(data)
            if mode != 1:
                output = m(data, decompose=True, sparsity=sparsity)
            loss = criterion(output, target)
            loss_value = loss.item()
            iter_loss.append(loss_value)
            test_loss += loss_value
            acc = compute_accuray(output, target)
            test_acc += acc
        test_losses.append(test_loss / len(test_loader))

        if mode != 2:
            print("Epoch {}: train loss is {}, train accuracy is {}; test loss is {}, test accuracy is {}".
                  format(epoch, round(train_loss / len(train_loader), 2),
                         round(train_acc / len(train_loader), 4),
                         round(test_loss / len(test_loader), 2),
                         round(test_acc / len(test_loader), 4)))
        if mode == 2:
            print("Epoch {}: test loss is {}, test accuracy is {}".
                  format(epoch,
                         round(test_loss / len(test_loader), 2),
                         round(test_acc / len(test_loader), 4)))

        if test_loss / len(test_loader) <= last_loss:
            save_model_path = os.path.join(out_dir, "models", "best_model.tar".format(epoch))
            torch.save({
                "model": m.state_dict(),
                "optimizer": optimizer.state_dict()
            }, save_model_path)
            last_loss = test_loss / len(test_loader)
            best_test_acc = round(test_acc / len(test_loader), 4)

    df = pd.DataFrame()
    df["iteration"] = np.arange(0, len(iter_loss))
    df["loss"] = iter_loss
    df.to_csv(iter_loss_path, index=False)

    df = pd.DataFrame()
    df["epoch"] = np.arange(0, nb_epochs)
    if mode != 2:
        df["train_loss"] = train_losses
    df["test_loss"] = test_losses
    df.to_csv(epoch_loss_path, index=False)

    if mode == 1:
        weights = {}
        for name, value in m.named_parameters():
            name1 = name.replace('.', '_')
            # print(name1, value)
            weights[name1] = m.cpu().state_dict()[name].numpy()
        weights['bn1_mean'] = m.bn1.running_mean.numpy()
        weights['bn1_var'] = m.bn1.running_var.numpy()
        weights['bn2_mean'] = m.bn2.running_mean.numpy()
        weights['bn2_var'] = m.bn2.running_var.numpy()
        io.savemat('weights_mnist.mat', weights)
        print('Finish training!')
        # torch.save(m.state_dict(), 'W.pth')
    print('The best test accuracy is:', best_test_acc)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savemat", dest="savemat", action="store_true")
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")
    parser.set_defaults(train=False)
    parser.set_defaults(decompose=False)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(test=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])), batch_size=128, shuffle=False
    )
    # print(len(train_loader))
    # print(len(test_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss().cuda()

    start = time.time()
    if args.savemat:
        net = LeNet().to(device)
        net.train(mode=True)
        train(net, "lenet", mode=1)

    if args.decompose or args.fine_tune:
        directory = 'mat_weights_mnist/rl3/'
        rpca_net = rpca_LeNet(load_dir=directory).to(device)  # changed according to compression rate
        # BN = io.loadmat('weight_mnist/BN.mat')
        weights = io.loadmat('weights_lenet5.mat')
        rpca_net.state_dict()['bn1.bias'].copy_(torch.from_numpy(weights['bn1_bias']).view(10))
        rpca_net.state_dict()['bn1.weight'].copy_(torch.from_numpy(weights['bn1_weight']).view(10))
        rpca_net.state_dict()['bn1.running_mean'].copy_(torch.from_numpy(weights['bn1_mean']).view(10))
        rpca_net.state_dict()['bn1.running_var'].copy_(torch.from_numpy(weights['bn1_var']).view(10))
        rpca_net.state_dict()['bn2.bias'].copy_(torch.from_numpy(weights['bn2_bias']).view(20))
        rpca_net.state_dict()['bn2.weight'].copy_(torch.from_numpy(weights['bn2_weight']).view(20))
        rpca_net.state_dict()['bn2.running_mean'].copy_(torch.from_numpy(weights['bn2_mean']).view(20))
        rpca_net.state_dict()['bn2.running_var'].copy_(torch.from_numpy(weights['bn2_var']).view(20))

        rpca_net.state_dict()['conv1.conv.weight'].copy_(torch.from_numpy(weights['conv1_weight']))
        rpca_net.state_dict()['conv2.conv.weight'].copy_(torch.from_numpy(weights['conv2_weight']))
        rpca_net.state_dict()['fc1.fc.weight'].copy_(torch.from_numpy(weights['fc1_weight']))
        rpca_net.state_dict()['fc1.fc.bias'].copy_(torch.from_numpy(weights['fc1_bias']).view(50))
        rpca_net.state_dict()['fc2.fc.weight'].copy_(torch.from_numpy(weights['fc2_weight']))
        rpca_net.state_dict()['fc2.fc.bias'].copy_(torch.from_numpy(weights['fc2_bias']).view(10))

        if args.decompose:
            sparsity = 1
            rpca_net.train(mode=False)
            train(rpca_net, "rpca_lenet", mode=2, sparsity=sparsity)  # changed according to sparsity
            print(directory)
            print('sparsity is', sparsity)
        else:
            sparsity = 0.99
            rpca_net.train(mode=True)
            train(rpca_net, "rpca_lenet_f", mode=3, sparsity=sparsity)  # changed according to sparsity
            print(directory)
            print('sparsity is', sparsity)

    elapsed = (time.time() - start)
    print("Finished. Time used:", elapsed)
