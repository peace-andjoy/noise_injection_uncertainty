# %%
import torch
import torch.nn as nn
from networks import FullyConnectedNN, NoisyFullyConnectedNN, FixNoiseFullyConnectedNN, DropoutFullyConnectedNN
import os
import pandas as pd
import numpy as np
from utils import *
import shutil
import argparse

parser = argparse.ArgumentParser(description='Training network for image classification',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_name', type=str, choices=['bostonHousing', 'concrete', 'energy', 'kin8nm', 'power-plant', 'protein-tertiary-structure', 'wine-quality-red', 'yacht'])
parser.add_argument('--model_name', type=str, choices=['FCN', 'noisyFCN', 'fixnoiseFCN', 'dropoutFCN'])
parser.add_argument('--init_noiselevel', type=float, default=0.1)
parser.add_argument('--fix_noiselevel', type=float, default=0.1)
parser.add_argument('--dropout_prob', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--optimizer_name', type=str, default="Adam")
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--epochs', type=int, default=50)

# %%
args = parser.parse_args()
dataset_name = args.dataset_name
model_name = args.model_name
if model_name == 'noisyFCN':
    init_noiselevel = args.init_noiselevel
elif model_name == 'fixnoiseFCN':
    fix_noiselevel = args.fix_noiselevel
elif model_name == 'dropoutFCN':
    dropout_prob = args.dropout_prob
batch_size = args.batch_size
optimizer_name = args.optimizer_name
learning_rate = args.learning_rate
weight_decay = args.weight_decay
epochs = args.epochs

X, y = load_uci_data_full(dataset_name)
eval_path = os.path.join(os.getcwd(), 'evaluations')
n_folds = 5
inverted_cv_fraction = 5
# Create eval dir for dataset
dataset_path = os.path.join(eval_path, dataset_name)
# Generate folds for this dataset
folds_path = os.path.join(dataset_path, 'fold_indices')
if dataset_name == 'protein-tertiary-structure':
    hidden_sizes = [1024, 1024, 1024, 1024]
else:
    hidden_sizes = [50, 50]
use_cuda = torch.cuda.is_available()

# Get dataset configuration
feature_indices, target_indices = load_uci_info(dataset_name)
input_size = len(feature_indices)
output_size = len(target_indices)

def def_optm(optimizer_name):
    global normal_param, learning_rate, momentum, weight_decay
    if optimizer_name == "SGD":
        print("using SGD as optimizer")
        optimizer = torch.optim.SGD([
                                    {'params': normal_param},
                                    {'params': alpha_param, 'weight_decay': 0}
                                    ],
                                    lr=learning_rate,
                                    momentum=momentum, weight_decay=weight_decay,
                                    nesterov=True)

    elif optimizer_name == "Adam":
        print("using Adam as optimizer")
        optimizer = torch.optim.Adam([
                                    {'params': normal_param},
                                    {'params': alpha_param, 'weight_decay': 0}
                                    ],
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        print("using RMSprop as optimizer")
        optimizer = torch.optim.RMSprop([
                                    {'params': normal_param},
                                    {'params': alpha_param, 'weight_decay': 0}
                                    ],
                                    lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=weight_decay, momentum=0)
    return optimizer

def def_model(model_name, input_size, hidden_sizes, output_size):
    print('model_name', model_name)
    if model_name == 'FCN':
        net = FullyConnectedNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    elif model_name == 'noisyFCN':
        net = NoisyFullyConnectedNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, init_noiselevel=init_noiselevel)
    elif model_name == 'fixnoiseFCN':
        net = FixNoiseFullyConnectedNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, fix_noiselevel=fix_noiselevel)
    elif model_name == 'dropoutFCN':
        net = DropoutFullyConnectedNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, dropout_prob=dropout_prob)
    return net

def train(model, train_loader, optimizer, criterion):
    model.train()
    losses = AverageMeter()
    for inputs, labels in train_loader:
        if use_cuda:
            labels = labels.cuda()
            inputs = inputs.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
    return losses.avg

def validate(model, val_loader, criterion):
    model.eval()
    if 'Dropout' in model._get_name():
        recursive_module_iteration(model)
    losses = AverageMeter()
    for inputs, labels in val_loader:
        if use_cuda:
            labels = labels.cuda()
            inputs = inputs.cuda()
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.update(loss.item(), inputs.size(0))
    return losses.avg

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename+'.pth.tar')
    if state['epoch'] % 10 == 0:
        torch.save(state, filename+'_epoch{}.pth.tar'.format(state['epoch']))
        last_filename = filename+'_epoch{}.pth.tar'.format(state['epoch']-10)
        if os.path.exists(last_filename):
            os.remove(last_filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_loss
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename+'.pth.tar', bestname)
        print("=> Obtain best accuracy, and update the best model")

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

def get_save_path():
    global model_name, fix_noiselevel, init_noiselevel, dropout_prob, dataset_path, batch_size, learning_rate, optimizer_name, fold
    if model_name == 'fixnoiseFCN':
        model_name2 = model_name + '_{}'.format(fix_noiselevel)
    elif model_name == 'noisyFCN':
        model_name2 = model_name + '_{}'.format(init_noiselevel)
    elif model_name == 'dropoutFCN':
        model_name2 = model_name + '_{}'.format(dropout_prob)
    else:
        model_name2 = model_name
    save_path = get_new_dir_in_parent_path(os.path.join(dataset_path, 'save_model_cv', model_name2+'_bs_{}_lr_{}_wd{}_optm_{}'.format(batch_size, learning_rate, weight_decay, optimizer_name), 'fold_{}'.format(fold)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

mean_val_loss = 0
mean_val_loss_best = 0
for fold in range(n_folds):
    print('fold {}'.format(fold))
    X_train, y_train, X_val, y_val = load_fold(folds_path, fold, X, y)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    val_data = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    net = def_model(model_name, input_size, hidden_sizes, output_size)
    normal_param = [
        param for name, param in net.named_parameters()
        if (not 'alpha_' in name and not 'alphafix_' in name)
    ] # this is the parameters do not contain noise scale coefficient

    alpha_param = [
        param for name, param in net.named_parameters()
        if 'alpha_' in name
    ]
    criterion = nn.MSELoss()
    if use_cuda:
        net.cuda()
        criterion.cuda()
    optimizer = def_optm(optimizer_name)
    best_loss = float('inf')
    save_path = get_save_path()

    for epoch in range(epochs):
        train_loss = train(model=net, train_loader=train_loader, optimizer=optimizer, criterion=criterion)
        print('epoch {}'.format(epoch))
        print('train_loss:', train_loss)
        val_loss = validate(model=net, val_loader=val_loader, criterion=criterion)
        print('val_loss:', val_loss)

        checkpoint_state = {
            'epoch': epoch + 1,
            'model_name': model_name,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        is_best = False
        if val_loss < best_loss:
            is_best = True
            best_loss = val_loss

        save_checkpoint(checkpoint_state, is_best,
                        save_path, 'checkpoint')
    mean_val_loss += val_loss
    mean_val_loss_best += best_loss
mean_val_loss /= n_folds
mean_val_loss_best /= n_folds
print('mean_val_loss:', mean_val_loss)
print('mean_val_loss_best:', mean_val_loss_best)

# %%
# 指定要写入的文件路径
file_path = 'results.txt'
if model_name == 'dropoutFCN':
    hyperparam_info = ', dropout_prob:{}'.format(dropout_prob)
elif model_name == 'fixnoiseFCN':
    hyperparam_info = ', fix_noiselevel:{}'.format(fix_noiselevel)
elif model_name == 'noisyFCN':
    hyperparam_info = ', init_noiselevel:{}'.format(init_noiselevel)
else:
    hyperparam_info = ''
res = 'datasets:{}, bs:{}, model:{}, optimizer:{}, lr:{}, weight_decay:{}, mean_val_loss:{}, mean_val_loss_best:{}'.format(dataset_name, batch_size, net._get_name(), optimizer_name, learning_rate, weight_decay, mean_val_loss, mean_val_loss_best) + hyperparam_info + '\n'
# 打开文件并写入内容
with open(file_path, 'a') as file:
    file.write(res)

# %%
print(net._get_name())

# %%
print([(name, param) for name, param in net.named_parameters() if 'alpha' in name])



    