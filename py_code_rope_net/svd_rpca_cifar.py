import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import scipy.io as io
import sys
import ResNet
from thop import profile
from torchtoolbox.transform import Cutout
from rpca_ResNet import *
import vgg
from vgg_rpca import *


model_names_res = sorted(name for name in ResNet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(ResNet.__dict__[name]))

model_names_vgg = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))

model_names = model_names_res + model_names_vgg


# print(model_names)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')  # resnet 200  vgg 300
parser.add_argument('--fepochs', default=50, type=int, metavar='N',
                    help='number of finetune epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')  # resnet 0.1  vgg 0.05
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')   # resnet 1e-4  vgg 5e-4
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')  # resnet 50  vgg 20
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--outlier', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_train_', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument("--save-mat", dest="save_mat", action="store_true")
parser.add_argument("--decompose", "--de", dest="decompose", action="store_true")
parser.add_argument("--fine_tune", "--fine", dest="fine_tune", action="store_true")
parser.add_argument("--svd", dest="svd", action="store_true")
parser.add_argument("--rpca", dest="rpca", action="store_true")
parser.add_argument('--sp', '--sparsity', default=1, type=float,
                    help='sparsity of E term')
parser.add_argument('--dataset', default='cifar10', type=str)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # print(args.arch)
    if args.decompose or args.fine_tune:
        if args.arch in model_names_res:
            directory = 'resnet110_r008/'  # changed according to compression rate and depth of resnet
            if args.dataset == 'cifar10':
                num_class = 10
            if args.dataset == 'cifar100':
                num_class = 100
            rpca_model = rpca_resnet110(load_dir='mat_weights_' + args.dataset + '/' + directory,
                                       num_classes=num_class).cuda()  # changed according to depth of resnet
            # if args.svd:
            #     rpca_model = rpca_resnet110(load_dir='pca_weights/', num_classes=num_class).cuda()
            num_blocks = 18  # changed according to depth of resnet, calculated as (depth-2)/6
            weights = io.loadmat('weights_resnet110_' + args.dataset + '.mat')   # changed according to depth of resnet
            rpca_model.state_dict()['bn1.bias'].copy_(torch.from_numpy(weights['bn1_bias']).view(16))
            rpca_model.state_dict()['bn1.weight'].copy_(torch.from_numpy(weights['bn1_weight']).view(16))
            rpca_model.state_dict()['bn1.running_mean'].copy_(torch.from_numpy(weights['bn1_mean']).view(16))
            rpca_model.state_dict()['bn1.running_var'].copy_(torch.from_numpy(weights['bn1_var']).view(16))
            rpca_model.state_dict()['conv1.conv.weight'].copy_(
                torch.from_numpy(weights['conv1_weight']))
            rpca_model.state_dict()['linear.fc.weight'].copy_(
                torch.from_numpy(weights['linear_weight']))
            if args.dataset == 'cifar10':
                rpca_model.state_dict()['linear.fc.bias'].copy_(
                    torch.from_numpy(weights['linear_bias']).view(10))
            if args.dataset == 'cifar100':
                rpca_model.state_dict()['linear.fc.bias'].copy_(
                    torch.from_numpy(weights['linear_bias']).view(100))

            for i in range(num_blocks):
                for j in range(1, 4):
                    rpca_model.state_dict()['layer' + str(j) + '.' + str(i) + '.bn1.bias'].copy_(
                        torch.from_numpy(weights['layer' + str(j) + '_' + str(i) + '_bn1_bias']).view(16 * 2**(j-1)))
                    rpca_model.state_dict()['layer' + str(j) + '.' + str(i) + '.bn1.weight'].copy_(
                        torch.from_numpy(weights['layer' + str(j) + '_' + str(i) + '_bn1_weight']).view(16 * 2**(j-1)))
                    rpca_model.state_dict()['layer' + str(j) + '.' + str(i) + '.bn1.running_mean'].copy_(
                        torch.from_numpy(weights[str(j) + '_' + str(i) + '_bn1_mean']).view(16 * 2**(j-1)))
                    rpca_model.state_dict()['layer' + str(j) + '.' + str(i) + '.bn1.running_var'].copy_(
                        torch.from_numpy(weights[str(j) + '_' + str(i) + '_bn1_var']).view(16 * 2**(j-1)))
                    rpca_model.state_dict()['layer' + str(j) + '.' + str(i) + '.bn2.bias'].copy_(
                        torch.from_numpy(weights['layer' + str(j) + '_' + str(i) + '_bn2_bias']).view(16 * 2**(j-1)))
                    rpca_model.state_dict()['layer' + str(j) + '.' + str(i) + '.bn2.weight'].copy_(
                        torch.from_numpy(weights['layer' + str(j) + '_' + str(i) + '_bn2_weight']).view(16 * 2**(j-1)))
                    rpca_model.state_dict()['layer' + str(j) + '.' + str(i) + '.bn2.running_mean'].copy_(
                        torch.from_numpy(weights[str(j) + '_' + str(i) + '_bn2_mean']).view(16 * 2**(j-1)))
                    rpca_model.state_dict()['layer' + str(j) + '.' + str(i) + '.bn2.running_var'].copy_(
                        torch.from_numpy(weights[str(j) + '_' + str(i) + '_bn2_var']).view(16 * 2**(j-1)))
                    rpca_model.state_dict()['layer' + str(j) + '.' + str(i) + '.conv1.conv.weight'].copy_(
                        torch.from_numpy(weights['layer' + str(j) + '_' + str(i) + '_conv1_weight']))
                    rpca_model.state_dict()['layer' + str(j) + '.' + str(i) + '.conv2.conv.weight'].copy_(
                        torch.from_numpy(weights['layer' + str(j) + '_' + str(i) + '_conv2_weight']))

        if args.arch in model_names_vgg:
            directory = 'vgg19_6/'  # changed according to compression rate and depth of vgg
            if args.sp == 1:
                sp_name = '1'
            if args.sp == 0.5:
                sp_name = '5'
            if args.sp == 0.6:
                sp_name = '6'
            if args.sp == 0.7:
                sp_name = '7'
            if args.sp == 0.8:
                sp_name = '8'
            if args.sp == 0.9:
                sp_name = '9'
            if args.sp == 0.95:
                sp_name = '95'
            if args.sp == 0.98:
                sp_name = '98'
            if args.sp == 0.99:
                sp_name = '99'
            log_path = 'rpca_result/' + args.dataset + '/vgg19/rank level6/E' \
                       + sp_name  # changed according to compression rate and depth of vgg
            if args.dataset == 'cifar10':
                num_class = 10
            if args.dataset == 'cifar100':
                num_class = 100
            rpca_model = rpca_vgg19(load_dir='mat_weights_' + args.dataset + '/' + directory,
                                    num_class=num_class).cuda()  # changed according to depth of vgg

        if args.decompose:
            log_path = log_path + 'd.txt'
            file = open(log_path, 'w')
            sys.stdout = file
            rpca_model.eval()
            validate(val_loader, rpca_model, criterion, sparsity=args.sp)

            print(directory)
            print('sparsity is', args.sp)
            input = torch.randn(1, 3, 32, 32).cuda()
            flops, params = profile(rpca_model, (input,))
            print('flops: ', flops, 'params: ', params)
            file.close()
            return

        if args.fine_tune:
            log_path = log_path + 'f.txt'
            file = open(log_path, 'w')
            sys.stdout = file
            sparsity = args.sp
            if args.outlier:
                directory = directory + 'outlier/'
            print(directory)
            print('sparsity is', sparsity)
            args.save_dir = 'save_fine_tune_' + args.dataset + '/' + directory + str(sparsity)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            rpca_model.train()
            if args.arch in model_names_res:
                optimizer = torch.optim.SGD(rpca_model.parameters(), args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40])
                if sparsity > 0.7:
                    args.fepochs = 100
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80])
            if args.arch in model_names_vgg:
                args.lr = args.lr * 0.2
                if args.arch == 'vgg16' or args.arch == 'vgg19':
                    args.lr = args.lr * 0.2
                optimizer = torch.optim.SGD(rpca_model.parameters(), args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)
                args.fepochs = 100
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.2, milestones=[40, 80])
            for epoch in range(args.fepochs):
                if args.arch in model_names_res:
                    if num_blocks == 18 and epoch == 0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = args.lr * 0.1
                if args.arch in model_names_vgg:
                    if epoch == 0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = args.lr * 0.1

                # train for one epoch
                print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
                train(train_loader, rpca_model, criterion, optimizer, epoch, sparsity=sparsity)
                lr_scheduler.step()
                if args.arch in model_names_res:
                    if num_blocks == 18 and epoch == 0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = args.lr

                if args.arch in model_names_vgg:
                    if epoch == 0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = args.lr

                # evaluate on validation set
                prec1 = validate(val_loader, rpca_model, criterion, sparsity=sparsity)


                # remember best prec@1 and save checkpoint
                if not args.svd:
                    is_best = prec1 > best_prec1
                    best_prec1 = max(prec1, best_prec1)

                    if epoch > 0 and epoch % args.save_every == 0:
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': rpca_model.state_dict(),
                            'best_prec1': best_prec1,
                        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

                    if is_best:
                        save_checkpoint({
                            'state_dict': rpca_model.state_dict(),
                            'best_prec1': best_prec1,
                        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

            input = torch.randn(1, 3, 32, 32).cuda()
            flops, params = profile(rpca_model, (input,))
            print('flops: ', flops, 'params: ', params)

            print(directory)
            print('sparsity is', sparsity)
            print('Finish fine_tune!')
            print('best Prec@1 is:', best_prec1)
            # for name, value in rpca_model.named_parameters():
            #     print(name, value)
            file.close()
            return

    args.save_dir = os.path.join(args.save_dir + args.dataset, args.arch)
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # model = torch.nn.DataParallel(ResNet.__dict__[args.arch]())
    if args.arch in model_names_res:
        if args.dataset == 'cifar10':
            model = ResNet.__dict__[args.arch]()
        if args.dataset == 'cifar100':
            model = ResNet.__dict__[args.arch](num_class=100)

    if args.arch in model_names_vgg:
        if args.dataset == 'cifar10':
            model = vgg.__dict__[args.arch]()
        if args.dataset == 'cifar100':
            model = vgg.__dict__[args.arch](num_class=100)
    model.cuda()
    # for name, value in model.named_parameters():
    #     print(name)
    # optionally resume from a checkpoint

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.outlier:
        if args.rpca:
            # model = rpca_resnet20(load_dir='mat_weights_' + args.dataset + '/' + 'resnet20_r008/',
            #                            num_classes=10).cuda()
            model = rpca_vgg16(load_dir='mat_weights_' + args.dataset + '/' + 'vgg16_6/', num_class=10).cuda()
        else:
            model = vgg.vgg16().cuda()
        if os.path.isfile(args.outlier):
            print("=> loading checkpoint '{}'".format(args.outlier))
            checkpoint = torch.load(args.outlier)
            # args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            print(best_prec1)
            model.load_state_dict(checkpoint['state_dict'])
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.evaluate, checkpoint['epoch']))
            sum = 0
            for i in range(10):
                val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                    Cutout(0.5),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=128, shuffle=False,
                num_workers=args.workers, pin_memory=True)
                if args.rpca:
                    sum += validate(val_loader, model, criterion, sparsity=0.99)
                else:
                    sum += validate(val_loader, model, criterion)
            avg = sum/10
            print('test acc is:', avg)
            return

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    if args.half:
        model.half()
        criterion.half()

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.arch in model_names_res:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 150], last_epoch=args.start_epoch - 1)
    if args.arch in model_names_vgg:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.5,
                                                            milestones=[30, 60, 90, 120, 150, 180, 210, 240, 270],
                                                            last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110', 'vgg19']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1



    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        if args.arch in ['resnet1202', 'resnet110', 'vgg19'] and epoch == 0:
            #  switch back to the normal lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        if is_best:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

    input = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(model, (input,))
    print('FLOPs: ', flops, 'params: ', params)

    print('Finish training!')
    print('best Prec@1 is:', best_prec1)

    if args.save_mat:
        if args.arch in model_names_res:
            if args.arch == 'resnet20':
                num_blocks = 3

            if args.arch == 'resnet32':
                num_blocks = 5

            if args.arch == 'resnet44':
                num_blocks = 7

            if args.arch == 'resnet56':
                num_blocks = 9

            if args.arch == 'resnet110':
                num_blocks = 18

            if args.arch == 'resnet1202':
                num_blocks = 200

            weights = {}
            for name, value in model.named_parameters():
                name1 = name.replace('.', '_')
                # print(name1, value)
                weights[name1] = model.cpu().state_dict()[name].numpy()
            weights['bn1_mean'] = model.bn1.running_mean.numpy()
            weights['bn1_var'] = model.bn1.running_var.numpy()

            # print((model.module.layer1[0].bn1.running_mean.numpy()))

            for i in range(num_blocks):
                weights['1_' + str(i) + '_bn1_mean'] = model.layer1[i].bn1.running_mean.numpy()
                weights['1_' + str(i) + '_bn1_var'] = model.layer1[i].bn1.running_var.numpy()
                weights['1_' + str(i) + '_bn2_mean'] = model.layer1[i].bn2.running_mean.numpy()
                weights['1_' + str(i) + '_bn2_var'] = model.layer1[i].bn2.running_var.numpy()

                weights['2_' + str(i) + '_bn1_mean'] = model.layer2[i].bn1.running_mean.numpy()
                weights['2_' + str(i) + '_bn1_var'] = model.layer2[i].bn1.running_var.numpy()
                weights['2_' + str(i) + '_bn2_mean'] = model.layer2[i].bn2.running_mean.numpy()
                weights['2_' + str(i) + '_bn2_var'] = model.layer2[i].bn2.running_var.numpy()

                weights['3_' + str(i) + '_bn1_mean'] = model.layer3[i].bn1.running_mean.numpy()
                weights['3_' + str(i) + '_bn1_var'] = model.layer3[i].bn1.running_var.numpy()
                weights['3_' + str(i) + '_bn2_mean'] = model.layer3[i].bn2.running_mean.numpy()
                weights['3_' + str(i) + '_bn2_var'] = model.layer3[i].bn2.running_var.numpy()

            io.savemat('weights_' + args.arch + '_' + args.dataset + '.mat', weights)

        if args.arch in model_names_vgg:
            weights = {}
            for name, value in model.named_parameters():
                name1 = name.replace('.', '_')
                # print(name1, value)
                weights[name1] = model.cpu().state_dict()[name].numpy()
            io.savemat('weights_' + args.arch + '_' + args.dataset + '.mat', weights)


def train(train_loader, model, criterion, optimizer, epoch, sparsity=1):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        if args.fine_tune:
            output = model(input_var, decompose=True, sparsity=sparsity)
        else:
            output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, sparsity=1):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            if args.decompose or args.fine_tune or args.rpca:
                output = model(input_var, decompose=True, sparsity=sparsity)
            else:
                output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
