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
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils_.utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models
import copy
import numpy as np
from models.attack_model import Attack
from models.nomarlization_layer import Normalize_layer, noise_Normalize_layer
from numpy.random import RandomState
import copy
import csv
from matplotlib import pyplot as plt
import re

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
args.use_cuda = False
args.ngpu = 1
args.manualSeed = 123
args.optimizer = 'SGD'
args.num_filters = 4

# %%
def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

# %%
res_path = '/home/xueqiong/noise_injection_uncertainty/cifar10/save/final_res_resnet8/'
folders = os.listdir(res_path)
# %%
for folder in folders:
    if folder == 'qualitative_analysis':
        continue
    args.folder = folder
    args.arch = re.search(r'cifar10_(.*?)_160', folder).group(1)
    if args.num_filters == 4:
        args.arch = re.search(r'cifar10_(.*?)_4', folder).group(1)
    if 'fixnoise' in folder:
        args.fix_noise_level = float(re.search(r'adv_(.*)', folder).group(1))
    args.resume_path = os.path.join(res_path, args.folder)
    args.save_path = os.path.join(args.resume_path, 'test')
    print('args.arch,', args.arch)
    print('args.resume_path,', args.resume_path)
    print('args.save_path,', args.save_path)

    if 'notadv' in folder:
        args.adv_train = False
    else:
        args.adv_train = True

    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path,
                            'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    if args.use_cuda == False:
        print_log('No GPU!', log)
        print_log('{},{}'.format(args.ngpu,torch.cuda.is_available()), log)
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(
        sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(
        torch.backends.cudnn.version()), log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log')

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

    # Init model, criterion, and optimizer
    if 'fixnoise' not in args.arch:
        net_c = models.__dict__[args.arch](num_classes, args.num_filters)
    else:
        net_c = models.__dict__[args.arch](num_classes, args.num_filters, args.fix_noise_level)
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

    print_log("=> network :\n {}".format(net), log)

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

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
        
    def validate(val_loader, model, criterion, log, test_times=10, return_detail=False):
        losses = AverageMeter()
        acc_avg = AverageMeter()
        # switch to evaluate mode
        samplesize = val_loader.dataset.data.shape[0]
        model.eval()
        if 'dropout' in args.arch:
            recursive_module_iteration(model)
        with torch.no_grad():
            var_softmaxs_all = []
            pred_all = []
            avg_logits_all = []
            for i, (input, target) in enumerate(val_loader):
                logits = torch.zeros((input.shape[0], num_classes, test_times))
                softmaxs = torch.zeros((input.shape[0], num_classes, test_times))
                if args.use_cuda:
                    target = target.cuda(non_blocking=True)
                    input = input.cuda()
                    logits = logits.cuda()
                for i in range(test_times):
                    # compute output
                    output = model(input)
                    logits[:, :, i] = output
                    softmaxs[:, :, i] = F.softmax(output, dim=1)
                avg_logits = torch.mean(logits, dim=2)
                var_softmaxs = torch.sum(torch.var(softmaxs, dim=2), dim=-1)   # [batch_size]
                var_softmaxs = var_softmaxs.numpy().tolist()
                loss = criterion(avg_logits, target)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(avg_logits.data, target, topk=(1, 5))
                acc_avg.update(prec1.item(), input.size(0))
                _, pred = output.topk(1, 1, True, True)
                pred = pred.t()[0].numpy().tolist()
                if return_detail:
                    var_softmaxs_all.extend(var_softmaxs)
                    pred_all.extend(pred)
                    avg_logits_all.extend(avg_logits)

        print_log(
        '  **Test** Prec@1 {top1.avg:.3f}'.format(top1=acc_avg), log)
        return acc_avg, var_softmaxs_all, pred_all, avg_logits_all

    for package in os.listdir(args.resume_path):
        if package == 'test' or '.npy' in package:
            continue
        resume = os.path.join(args.resume_path, package, 'checkpoint_epoch160.pth.tar')
        print_log("=> loading checkpoint '{}'".format(resume), log)
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

        print_log("=> loaded checkpoint '{}' (epoch {})".format(
            resume, args.start_epoch), log)
        cur_acc, var_softmaxs_all_all, pred_all, avg_logits_all = validate(test_loader, net, criterion, log, 10, True)
        break

    print(len(var_softmaxs_all_all))
    print(len(pred_all))

    sorted_index = np.argsort(var_softmaxs_all_all)
    total_num = len(sorted_index)
    y_pre_array = np.array(avg_logits_all)
    y_pre_label_array = np.array(pred_all)
    y_test = np.array(test_data.targets)
    sorted_y_pred = y_pre_array[sorted_index]  # 从预测方差从小到大排序
    sorted_y_data = y_test[sorted_index]
    loss_list = []
    acc_list = []

    for i in range(1, total_num + 1):
        sorted_y_pred1 = sorted_y_pred[0:i]
        sorted_y_data1 = sorted_y_data[0:i]
        loss = criterion(torch.tensor(sorted_y_pred1), torch.tensor(sorted_y_data1))
        loss = float(loss)
        acc = np.mean(sorted_y_data1 == np.argmax(sorted_y_pred1, axis=1))
        print(acc)
        loss_list.append(loss)
        acc_list.append(acc)

    sorted_index = np.argsort(var_softmaxs_all_all)
    total_num = len(sorted_index)

    # 平均输出的loss排序
    sorted_loss = np.array(loss_list)[sorted_index]
    epochs = np.arange(total_num)

    print(acc_list)
    print(loss_list)

    np.save(os.path.join(args.resume_path, 'acc_list.npy'), acc_list)
    np.save(os.path.join(args.resume_path, 'loss_list.npy'), loss_list)
# %%
def get_fig_label(model_name):
    if 'dropout' in model_name:
        return "MC dropout"
    elif 'fixnoise' in model_name:
        return "MCNI (fixed level)"
    elif "noise_" in model_name:
        return "MCNI (learned level)"
# %%
# 画图
total_num = 10000
epochs = np.arange(total_num)
folders = os.listdir(res_path)
folders.sort()
plt.figure(dpi=300)
for folder in folders:
    if folder == 'qualitative_analysis':
        continue
    if 'notadv' not in folder or 'vanilla' in folder:
        continue
    if 'fixnoise' in folder and ('0.2' in folder or '0.5' in folder or '0.1' in folder):
        continue
    elif 'noise' not in folder and 'dropout' not in folder:
        continue
    args.folder = folder
    args.arch = re.search(r'cifar10_(.*?)_160', folder).group(1)
    if args.num_filters == 4:
        args.arch = re.search(r'cifar10_(.*?)_4', folder).group(1)
    label = args.arch
    if 'fixnoise' in folder:
        args.fix_noise_level = float(re.search(r'adv_(.*)', folder).group(1))
        label = label + '_{}'.format(args.fix_noise_level)
    args.resume_path = os.path.join(res_path, args.folder)
    args.save_path = os.path.join(args.resume_path, 'test')
    loss_list = np.load(os.path.join(args.resume_path, 'loss_list.npy'))
    acc_list = np.load(os.path.join(args.resume_path, 'acc_list.npy'))
    # 根据输出方差排序样本后，前n个样本输出均值的平均accuracy
    plt.plot((epochs + 1)/len(epochs), acc_list, label=get_fig_label(folder), linestyle='--')  # bo:blue dot蓝点
plt.legend()
plt.xlabel('Coverage', fontsize=16)
plt.ylabel('Risk(Acc)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0.05)
plt.show()
# %%
total_num = 10000
epochs = np.arange(total_num)
folders = os.listdir(res_path)
plt.figure(dpi=300)
for folder in folders:
    if folder == 'qualitative_analysis':
        continue
    if 'notadv' not in folder or 'vanilla' in folder:
        continue
    args.folder = folder
    args.arch = re.search(r'cifar10_(.*?)_160', folder).group(1)
    if args.num_filters == 4:
        args.arch = re.search(r'cifar10_(.*?)_4', folder).group(1)
    if 'fixnoise' in folder:
        args.fix_noise_level = float(re.search(r'adv_(.*)', folder).group(1))
    args.resume_path = os.path.join(res_path, args.folder)
    args.save_path = os.path.join(args.resume_path, 'test')
    loss_list = np.load(os.path.join(args.resume_path, 'loss_list.npy'))
    acc_list = np.load(os.path.join(args.resume_path, 'acc_list.npy'))
    # 根据输出方差排序样本后，前n个样本输出均值的平均accuracy
    plt.plot((epochs + 1)/len(epochs), loss_list, label=folder, linestyle='--')  # bo:blue dot蓝点
plt.legend()
plt.xlabel('Coverage', fontsize=16)
plt.ylabel('Risk(Loss)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
# %%
