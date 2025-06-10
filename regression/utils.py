import os

import pandas as pd
import numpy as np

from helper import make_path_if_missing
from helper import get_new_dir_in_parent_path

import torch
import torch.nn as nn
from networks import FullyConnectedNN, NoisyFullyConnectedNN, FixNoiseFullyConnectedNN, DropoutFullyConnectedNN

import logging

DATA_PATH = "/home/xueqiong/noise_injection_uncertainty/regression/data/"

def load_uci_info(name):
    dataset_path = os.path.join(DATA_PATH, name)

    feature_indices_path = os.path.join(dataset_path, 'index_features.txt')
    feature_indices = np.loadtxt(feature_indices_path, dtype=int).tolist()

    target_indices_path = os.path.join(dataset_path, 'index_target.txt')
    target_indices = np.loadtxt(target_indices_path, dtype=int).tolist()

    if type(feature_indices) == int:
        feature_indices = [feature_indices]
    if type(target_indices) == int:
        target_indices = [target_indices]

    return feature_indices, target_indices


def load_uci_data_as_dataframe(name):
    dataset_path = os.path.join(DATA_PATH, name)
    all_files = os.listdir(dataset_path)
    dataset_filename = next(f for f in all_files if f.endswith('.data'))
    dataset_path = os.path.join(dataset_path, dataset_filename)
    return pd.read_csv(dataset_path, engine='python', header=None, delim_whitespace=True)


def load_uci_data_full(name):
    df = load_uci_data_as_dataframe(name)

    feature_indices, target_indices = load_uci_info(name)
    X = df.loc[:, feature_indices].values
    y = df.loc[:, target_indices].values

    # Ensure target arrays are 2 dimensional
    y = y.reshape(-1, 1) if len(y.shape) == 1 else y
    return X, y


def load_fold(folds_path, fold, X_full, y_full):
    train_idx_path = os.path.join(folds_path, '{}_train_idx.txt'.format(fold))
    val_idx_path = os.path.join(folds_path, '{}_val_idx.txt'.format(fold))
    train_idx = np.loadtxt(train_idx_path, dtype=int).tolist()
    val_idx = np.loadtxt(val_idx_path, dtype=int).tolist()

    X_train = X_full[train_idx, :]
    y_train = y_full[train_idx, :]
    X_val = X_full[val_idx, :]
    y_val = y_full[val_idx, :]

    return X_train, y_train, X_val, y_val


def load_uci_data_test(dataset_name):
    X_full, y_full = load_uci_data_full(dataset_name)
    indices_path = os.path.join(DATA_PATH, dataset_name, 'train_cv-test')

    train_idx_path = os.path.join(indices_path, 'train_cv_indices.txt')
    test_idx_path = os.path.join(indices_path, 'test_indices.txt')

    train_idx = np.loadtxt(train_idx_path, dtype=int).tolist()
    test_idx = np.loadtxt(test_idx_path, dtype=int).tolist()

    X_train = X_full[train_idx, :]
    y_train = y_full[train_idx, :]
    X_test = X_full[test_idx, :]
    y_test = y_full[test_idx, :]

    return X_train, y_train, X_test, y_test

def save_indices(path, filename, indices):
    make_path_if_missing(path)
    file_path = os.path.join(path, filename)
    with open(file_path, 'w') as f:
        np.savetxt(f, indices, fmt='%d')


def create_folds(dataset_name, n_folds, inverted_cv_fraction, parent_path):
    train_cv_idx_path = os.path.join(DATA_PATH, dataset_name, 'train_cv-test', 'train_cv_indices.txt')
    train_cv_idx = np.loadtxt(train_cv_idx_path, dtype=int).tolist()
    np.random.shuffle(train_cv_idx)

    val_idx_per_fold = np.array_split(train_cv_idx, inverted_cv_fraction)[:n_folds]
    folds_path = get_new_dir_in_parent_path(parent_path, subdir='fold_indices')

    for i, fold_val_idx in enumerate(val_idx_per_fold):
        # Get (sorted) fold_train_idx and shuffle
        fold_train_idx = np.setdiff1d(train_cv_idx, fold_val_idx)
        np.random.shuffle(fold_train_idx)

        # Save fold
        save_indices(folds_path, '{}_val_idx.txt'.format(i), fold_val_idx)
        save_indices(folds_path, '{}_train_idx.txt'.format(i), fold_train_idx)

    return folds_path

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

def def_model(model_name, input_size, hidden_sizes, output_size, init_noiselevel=None, fix_noiselevel=None, dropout_prob=None):
    print('model_name', model_name)
    if model_name == 'FCN':
        net = FullyConnectedNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    elif model_name == 'noisyFCN':
        assert init_noiselevel
        net = NoisyFullyConnectedNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, init_noiselevel=init_noiselevel)
    elif model_name == 'fixnoiseFCN':
        assert fix_noiselevel
        net = FixNoiseFullyConnectedNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, fix_noiselevel=fix_noiselevel)
    elif model_name == 'dropoutFCN':
        assert dropout_prob
        net = DropoutFullyConnectedNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, dropout_prob=dropout_prob)
    return net

def mean_standardized_log_loss(
    y_true, y_pred, y_std, sample_weight=None, multioutput="uniform_average", squared=True
):
    """Mean standardized log loss.
    Read more in the :ref:`User Guide <mean_standardized_log_loss>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    y_std : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated standard deviation in predictions.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    Examples
    --------
    >>> from sklearn.metrics import mean_standardized_log_loss
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> y_std = [0.1, 0, 0.05, 0.3]
    >>> mean_standardized_log_loss(y_true, y_pred, y_std)
    6.356
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> y_std = [[0.01, 0.02],[0.01,0.04],[0.03,0.04]]
    >>> mean_standardized_log_loss(y_true, y_pred, y_std)
    5.511
    >>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
    array([5.00107605, 6.02159874])
    >>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    2.858
    """
    # y_type, y_true, y_pred, multioutput = _check_reg_targets(
    #   y_true, y_pred, multioutput
    # )
    # check_consistent_length(y_true, y_pred, sample_weight)
    
    ###########
    # Checks like the above ones to be implemented.
    ###########
    
    first_term = 0.5 * np.log(2 * np.pi * y_std**2)
    second_term = ((y_true - y_pred)**2)/(2 * y_std**2)
    
    output_errors = np.average(first_term + second_term, axis=0, weights=sample_weight)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    print('first_term', np.average(np.average(first_term, axis=0, weights=sample_weight), weights=multioutput))
    print('second_term', np.average(np.average(second_term, axis=0, weights=sample_weight), weights=multioutput))

    return np.average(output_errors, weights=multioutput)

def validate(model, val_loader, criterion, test_times=100, use_cuda=False):
    model.eval()
    if 'Dropout' in model._get_name():
        recursive_module_iteration(model)
    losses = AverageMeter()
    mslls = AverageMeter()
    with torch.no_grad():
        var_outputs_all = []
        pred_all = []
        for inputs, labels in val_loader:
            outputs = torch.zeros((inputs.shape[0], 1, test_times))
            if use_cuda:
                labels = labels.cuda()
                inputs = inputs.cuda()
                outputs = outputs.cuda()
            # compute output
            for i in range(test_times):
                output = model(inputs)
                outputs[:, :, i] = output
            avg_outputs = torch.mean(outputs, dim=2)
            var_outputs = torch.sum(torch.var(outputs, dim=2), dim=-1)
            loss = criterion(avg_outputs, labels)
            losses.update(loss.item(), inputs.size(0))
            msll = mean_standardized_log_loss(labels, avg_outputs, torch.sqrt(var_outputs))
            mslls.update(msll, inputs.size(0))
            if use_cuda:
                var_outputs = var_outputs.cpu()
                avg_outputs = avg_outputs.cpu()
            var_outputs = var_outputs.numpy().tolist()
            var_outputs_all.extend(var_outputs)
            avg_outputs = avg_outputs.numpy().tolist()
            pred_all.extend(avg_outputs)
            
    return losses.avg, var_outputs_all, pred_all, mslls.avg

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

def def_optm(optimizer_name, normal_param, alpha_param, learning_rate, momentum=0, weight_decay=None):
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

def train(model, train_loader, optimizer, criterion, use_cuda):
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

def get_logger():
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename='evaluation.log', mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger