# %%
import torch
import torch.nn as nn
from networks import FullyConnectedNN, NoisyFullyConnectedNN, FixNoiseFullyConnectedNN, DropoutFullyConnectedNN
import os
import pandas as pd
import numpy as np
from utils import *
import shutil
import itertools
from config import configs

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# %%
res = []
dataset_names = configs['datasets']
model_names = configs['models']
learning_rates = configs['learning_rates']
weight_decays = configs['weight_decays']
optimizer_name = configs['optimizer_name']
hyperparam_eval_interval = configs['hyperparam_eval_interval']
patience = configs['patience']
global_max_epochs = configs['global_max_epochs']
dropout_probs = configs['dropout_probs']
fix_noiselevels = configs['fix_noiselevels']
init_noiselevels = configs['init_noiselevels']
dropout_batch_size = configs['dropout_batch_size']
n_hidden = configs['n_hidden']
nonlinearity = configs['nonlinearity']
test_times = configs['mc_samples']
normalize_X = configs['normalize_X']
normalize_y = configs['normalize_y']
n_folds = configs['n_folds']
inverted_cv_fraction = configs['inverted_cv_fraction']
use_cuda = torch.cuda.is_available()
print('use_cuda:', use_cuda)
logger = get_logger()

# %%
def normalize(X_train, y_train, normalize_X=True, normalize_y=False):
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)

    # Set std dev to 1 for constant features
    X_std[np.all(X_train == X_train[0, :], axis=0)] = 1.0

    X_train = (X_train - X_mean) / X_std if normalize_X else X_train
    y_train = (y_train - y_mean) / y_std if normalize_y else y_train
    return X_train, y_train

def cross_validation(X_full, y_full, model_name, input_size, hidden_sizes, output_size, init_noiselevel, fix_noiselevel, dropout_prob, batch_size, normalize_X, normalize_y, weight_decay):
    # 交叉验证。output：交叉验证的loss
    criterion = nn.MSELoss()
    # 存放n_folds个dataloader和model、optimizer
    train_loaders = []
    val_loaders = []
    models = []
    optimizers = []
    for fold in range(n_folds):
        print('fold {}'.format(fold))
        # 准备好数据集
        X_train, y_train, X_val, y_val = load_fold(folds_path, fold, X_full, y_full)
        X_train, y_train = normalize(X_train, y_train, normalize_X, normalize_y)
        X_val, y_val = normalize(X_val, y_val, normalize_X, normalize_y)
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)

        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        val_data = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

        # 创建模型
        model = def_model(model_name, input_size, hidden_sizes, output_size, init_noiselevel, fix_noiselevel, dropout_prob)
        models.append(model)

        normal_param = [
            param for name, param in model.named_parameters()
            if (not 'alpha_' in name and not 'alphafix_' in name)
        ] # this is the parameters do not contain noise scale coefficient
        alpha_param = [
            param for name, param in model.named_parameters()
            if 'alpha_' in name
        ]
        
        if use_cuda:
            model.cuda()
            criterion.cuda()
        optimizer = def_optm(optimizer_name, normal_param, alpha_param, learning_rate, weight_decay=weight_decay)
        optimizers.append(optimizer)

    # 用于记录epoch和loss
    epoch2loss = {}
    best_epoch = 0
    best_loss = float('inf')
    no_improve_num = 0
    # 开始训练
    for epoch in range(global_max_epochs):
        train_losses = []   # 记录5个模型的train_loss
        val_losses = []
        for model, train_loader, val_loader, optimizer in zip(models, train_loaders, val_loaders, optimizers):
            train_loss = train(model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion, use_cuda=use_cuda)
            train_losses.append(train_loss)
            if epoch % hyperparam_eval_interval == 0:
                val_loss, _, _ = validate(model, val_loader, criterion, test_times, use_cuda)
                val_losses.append(val_loss)
        if epoch % hyperparam_eval_interval == 0:
            avg_val_loss = np.mean(val_losses)
            epoch2loss[epoch] = avg_val_loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch
                no_improve_num = 0
            else:
                no_improve_num += 1
                print('no_improve_num: {}'.format(no_improve_num))
        print('epoch {}'.format(epoch))
        print('avg_train_loss: {:.4f}, train_losses:{}'.format(np.mean(train_losses), train_losses))
        if val_losses:
            print('avg_val_loss: {:.4f}, val_losses:{}'.format(np.mean(val_losses), val_losses))
        if no_improve_num >= patience:
            break
    return epoch2loss, best_epoch, best_loss

# %%
result_df = []
for dataset_name in dataset_names:
    # Get dataset configuration
    feature_indices, target_indices = load_uci_info(dataset_name)
    input_size = len(feature_indices)
    output_size = len(target_indices)
    batch_sizes = configs['batch_sizes_specific'].get('energy',  configs['batch_sizes'])
    num_units = configs['num_units_specific'].get('protein-tertiary-structure',  configs['num_units'])
    hidden_sizes = [num_units] * n_hidden
    X_full, y_full = load_uci_data_full(dataset_name)

    init_noiselevel, fix_noiselevel, dropout_prob = 0.01, 0.01, 0.01
    for model_name in model_names:
        eval_path = os.path.join(os.getcwd(), 'evaluations')
        # Create eval dir for dataset
        dataset_path = os.path.join(eval_path, dataset_name)
        # Generate folds for this dataset
        folds_path = os.path.join(dataset_path, 'fold_indices')
        for batch_size in batch_sizes:
            for weight_decay in weight_decays:
                for learning_rate in learning_rates:
                    if model_name == 'fixnoiseFCN':
                        for fix_noiselevel in fix_noiselevels:
                            print('batch_size:{}, weight_decay:{}, learning_rate:{}, model_name:{}, fix_noiselevel:{}'.format(batch_size, weight_decay, learning_rate, model_name, fix_noiselevel))
                            epoch2loss, best_epoch, best_loss = cross_validation(X_full, y_full, model_name, input_size, hidden_sizes, output_size, init_noiselevel, fix_noiselevel, dropout_prob, batch_size, normalize_X, normalize_y, weight_decay)
                            line = (dataset_name, model_name, batch_size, weight_decay, learning_rate, epoch2loss, best_epoch, best_loss, fix_noiselevel, init_noiselevel, dropout_prob)
                            logger.info(line)
                            result_df.append(line)
                    elif model_name == 'noisyFCN':
                        for init_noiselevel in init_noiselevels:
                            print('batch_size:{}, weight_decay:{}, learning_rate:{}, model_name:{}, init_noiselevel:{}'.format(batch_size, weight_decay, learning_rate, model_name, init_noiselevel))
                            epoch2loss, best_epoch, best_loss = cross_validation(X_full, y_full, model_name, input_size, hidden_sizes, output_size, init_noiselevel, fix_noiselevel, dropout_prob, batch_size, normalize_X, normalize_y, weight_decay)
                            line = (dataset_name, model_name, batch_size, weight_decay, learning_rate, epoch2loss, best_epoch, best_loss, fix_noiselevel, init_noiselevel, dropout_prob)
                            logger.info(line)
                            result_df.append(line)
                    elif model_name == 'dropoutFCN':
                        for dropout_prob in dropout_probs:
                            print('batch_size:{}, weight_decay:{}, learning_rate:{}, model_name:{}, dropout_prob:{}'.format(batch_size, weight_decay, learning_rate, model_name, dropout_prob))
                            epoch2loss, best_epoch, best_loss = cross_validation(X_full, y_full, model_name, input_size, hidden_sizes, output_size, init_noiselevel, fix_noiselevel, dropout_prob, batch_size, normalize_X, normalize_y, weight_decay)
                            line = (dataset_name, model_name, batch_size, weight_decay, learning_rate, epoch2loss, best_epoch, best_loss, fix_noiselevel, init_noiselevel, dropout_prob)
                            logger.info(line)
                            result_df.append(line)
                    elif model_name == 'FCN':
                        print('batch_size:{}, weight_decay:{}, learning_rate:{}, model_name:{}'.format(batch_size, weight_decay, learning_rate, model_name))
                        epoch2loss, best_epoch, best_loss = cross_validation(X_full, y_full, model_name, input_size, hidden_sizes, output_size, init_noiselevel, fix_noiselevel, dropout_prob, batch_size, normalize_X, normalize_y, weight_decay)
                        line = (dataset_name, model_name, batch_size, weight_decay, learning_rate, epoch2loss, best_epoch, best_loss, fix_noiselevel, init_noiselevel, dropout_prob)
                        logger.info(line)
                        result_df.append(line)

# %%
result_df = pd.DataFrame(result_df, columns=['dataset_name', 'model_name', 'batch_size', 'weight_decay', 'learning_rate', 'epoch2loss', 'best_epoch', 'best_loss', 'fix_noiselevel', 'init_noiselevel', 'dropout_prob'])
print(result_df)
from datetime import datetime
# 获取当前日期和时间
current_datetime = datetime.now()
# 格式化为字符串
current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
result_df.to_csv('predictive_res_{}.csv'.format(current_datetime_str), index=False)