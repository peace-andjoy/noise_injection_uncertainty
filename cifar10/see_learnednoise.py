# %%
import os
import sys
import shutil
import time
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils_.utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from tensorboardX import SummaryWriter
import models
import copy
import numpy as np
from models.attack_model import Attack
from models.nomarlization_layer import Normalize_layer, noise_Normalize_layer
from numpy.random import RandomState
import copy
from datetime import datetime

# %%
from argparse import Namespace
# 创建一个命名空间对象
args = Namespace()
args.dataset = 'cifar10'
args.adv_eval = False
args.data_path = '/home/xueqiong/CVPR_2019_PNI-master/data'
args.batch_size = 500
args.workers = 4
args.input_noise = False
args.learning_rate = 0.1
args.momentum = 0.9
args.decay = 0.0003
args.fine_tune = False
# -------------上面的基本上不用动，下面的要改
args.use_cuda = True
args.arch = 'noise_resnet8'
args.adv_train = False
args.fix_noise_level = 0.1
num_filters = 4
args.ngpu = 1
args.folder = "cifar10_noise_resnet8_4_160_SGD_notadv"
args.resume_path = '/home/xueqiong/noise_injection_uncertainty/cifar10/save/final_res_resnet8/{}'.format(args.folder)
args.save_path = os.path.join(args.resume_path, 'test')
args.manualSeed = 123
args.optimizer = 'SGD'

# %%
# Init logger
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
state = {k: v for k, v in args._get_kwargs()}
# Init the tensorboard path and writer
tb_path = os.path.join(args.save_path, 'tb_log')
writer = SummaryWriter(tb_path)

# %%
if args.dataset == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
elif args.dataset == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
elif args.dataset == 'svhn':
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
elif args.dataset == 'mnist':
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
elif args.dataset == 'imagenet':
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
else:
    assert False, "Unknow dataset : {}".format(args.dataset)

# Current data-preprocessing does not include the normalization
imagenet_train_transform = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()]
imagenet_test_transform = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()]

normal_train_transform = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()]
normal_test_transform = [
    transforms.ToTensor()]

# if not performing the adversarial training or evalutaion, we append
# the normalization back to the preprocessing
if not (args.adv_train or args.adv_eval):
    imagenet_train_transform.append(transforms.Normalize(mean, std))
    imagenet_test_transform.append(transforms.Normalize(mean, std))
    normal_train_transform.append(transforms.Normalize(mean, std))
    normal_test_transform.append(transforms.Normalize(mean, std))

if args.dataset == 'imagenet':
    train_transform = transforms.Compose(imagenet_train_transform)
    test_transform = transforms.Compose(imagenet_test_transform)
else:
    train_transform = transforms.Compose(normal_train_transform)
    test_transform = transforms.Compose(normal_test_transform)

if args.dataset == 'mnist':
    train_data = dset.MNIST(
        args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.MNIST(args.data_path, train=False,
                            transform=test_transform, download=True)
    num_classes = 10
elif args.dataset == 'cifar10':
    train_data = dset.CIFAR10(
        args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(
        args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10
elif args.dataset == 'cifar100':
    train_data = dset.CIFAR100(
        args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100(
        args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 100
elif args.dataset == 'svhn':
    train_data = dset.SVHN(args.data_path, split='train',
                            transform=train_transform, download=True)
    test_data = dset.SVHN(args.data_path, split='test',
                            transform=test_transform, download=True)
    num_classes = 10
elif args.dataset == 'stl10':
    train_data = dset.STL10(
        args.data_path, split='train', transform=train_transform, download=True)
    test_data = dset.STL10(args.data_path, split='test',
                            transform=test_transform, download=True)
    num_classes = 10
elif args.dataset == 'imagenet':
    train_dir = os.path.join(args.data_path, 'train')
    test_dir = os.path.join(args.data_path, 'val')
    train_data = dset.ImageFolder(train_dir, transform=train_transform)
    test_data = dset.ImageFolder(test_dir, transform=test_transform)
    num_classes = 1000
else:
    assert False, 'Do not support dataset : {}'.format(args.dataset)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.workers, pin_memory=False)

# %%
# Init model, criterion, and optimizer
if 'fixnoise' not in args.arch:
    net_c = models.__dict__[args.arch](num_classes, num_filters)
else:
    net_c = models.__dict__[args.arch](num_classes, num_filters, args.fix_noise_level,)
        # For adversarial case, override the original network with normalization layer
if (args.adv_train or args.adv_eval):
    if not args.input_noise:
        net = torch.nn.Sequential(
                Normalize_layer(mean,std),
                net_c
                )
    else:
        net = torch.nn.Sequential(
                noise_Normalize_layer(mean,std),
                net_c
                )           
else:
    net = net_c

# %%
if args.use_cuda:
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

# define loss function (criterion) and optimizer
criterion = torch.nn.CrossEntropyLoss()

if args.use_cuda:
    net.cuda()
    criterion.cuda()

# %%
def recursive_module_iteration(module, depth=0):
    # 打印当前模块的信息
    print(f"{' ' * (depth * 2)}Module: {module.__class__.__name__}")

    # 如果当前模块是容器模块（如 Sequential），则递归遍历其子模块
    if hasattr(module, "children"):
        for child in module.children():
            recursive_module_iteration(child, depth + 1)
    if isinstance(module, nn.Dropout):
        module.train()
        print('dropout设置为训练模式')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res
    
def cal_val_time(input, target, model, test_times=10):
    losses = AverageMeter()
    acc_avg = AverageMeter()
    model.eval()
    if 'dropout' in args.arch:
        recursive_module_iteration(model)
    logits = torch.zeros((input.shape[0], num_classes, test_times))
    if args.use_cuda:
        target = target.cuda(non_blocking=True)
        input = input.cuda()
        logits = logits.cuda()
    for i in range(test_times):
        # compute output
        output = model(input)

# %% [markdown]
# # predictive performance

# %%
package = '/home/xueqiong/noise_injection_uncertainty/cifar10/save/final_res_resnet8/cifar10_noise_resnet8_4_160_SGD_notadv/321'

# %%
resume = os.path.join(args.resume_path, package, 'checkpoint_epoch160.pth.tar')
if args.use_cuda == True:
    checkpoint = torch.load(resume)
else:
    checkpoint = torch.load(resume, map_location='cpu')
if not (args.fine_tune):
    args.start_epoch = checkpoint['epoch']
    recorder = checkpoint['recorder']

state_tmp = net.state_dict()
if 'state_dict' in checkpoint.keys():
    state_tmp.update(checkpoint['state_dict'])
else:
    state_tmp.update(checkpoint)

net.load_state_dict(state_tmp)
# %%
print([net.state_dict()[k] for k in net.state_dict().keys() if 'alpha' in k])
