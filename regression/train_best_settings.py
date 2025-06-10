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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_cuda = torch.cuda.is_available()
print('use_cuda:', use_cuda)

# %%
dropout_batch_size = configs['dropout_batch_size']
n_hidden = configs['n_hidden']
nonlinearity = configs['nonlinearity']
test_times = configs['mc_samples']
normalize_X = configs['normalize_X']
normalize_y = configs['normalize_y']
optimizer_name = configs['optimizer_name']
hyperparam_eval_interval = configs['hyperparam_eval_interval']

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

def get_logger2():
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename='train_best_settings.log', mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

# %%
# for dataset_name in ['bostonHousing', 'energy', 'wine-quality-red', 'power-plant',
#        'yacht', 'kin8nm', 'concrete', 'protein-tertiary-structure']:
for dataset_name in ['bostonHousing']:
    # for model_name in ['FCN', 'noisyFCN', 'fixnoiseFCN', 'dropoutFCN']:
    # for model_name in ['noisyFCN', 'fixnoiseFCN']:
    for model_name in ['fixnoiseFCN']:
        result_df = pd.read_csv("settings_loss.csv", index_col=0)
        result_df = result_df[result_df['batch_size']==32]
        result_df = result_df.loc[result_df.groupby(['dataset_name', 'model_name'])['best_loss'].idxmin()]

        result_df = result_df[result_df['dataset_name']==dataset_name]
        result_df = result_df[result_df['model_name']==model_name]
        for repeat_time in range(5):
            logger = get_logger2()
            for dataset_name, model_name, batch_size, weight_decay, learning_rate, fix_noiselevel, init_noiselevel, dropout_prob, best_epochs in result_df[['dataset_name', 'model_name', 'batch_size', 'weight_decay', 'learning_rate', 'fix_noiselevel', 'init_noiselevel', 'dropout_prob', 'best_epoch']].values:
                eval_path = os.path.join(os.getcwd(), 'evaluations')
                # Create eval dir for dataset
                dataset_path = os.path.join(eval_path, dataset_name)
                if dataset_name == 'protein-tertiary-structure':
                    hidden_sizes = [100, 100]
                else:
                    hidden_sizes = [50, 50]
                use_cuda = torch.cuda.is_available()


                # 准备好数据集
                X_train, y_train, X_test, y_test = load_uci_data_test(dataset_name)
                # Get dataset configuration
                feature_indices, target_indices = load_uci_info(dataset_name)
                input_size = len(feature_indices)
                output_size = len(target_indices)
                print(input_size, output_size)
                print(len(X_train), len(X_test), len(X_train)+len(X_test))

                X_train, y_train = normalize(X_train, y_train, normalize_X, normalize_y)
                X_test, y_test = normalize(X_test, y_test, normalize_X, normalize_y)
                
                X_train = torch.tensor(X_train, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32)
                train_data = torch.utils.data.TensorDataset(X_train, y_train)
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)

                X_test = torch.tensor(X_test, dtype=torch.float32)
                y_test = torch.tensor(y_test, dtype=torch.float32)
                test_data = torch.utils.data.TensorDataset(X_test, y_test)
                test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

                # 创建模型
                model = def_model(model_name, input_size, hidden_sizes, output_size, init_noiselevel, fix_noiselevel, dropout_prob)
                criterion = nn.MSELoss()

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

                # 用于记录epoch和loss
                epoch2loss_train = {}
                epoch2loss_test = {}
                best_epoch = 0
                best_loss = float('inf')
                no_improve_num = 0
                # 开始训练
                for epoch in range(best_epochs):
                    train_loss = train(model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion, use_cuda=use_cuda)
                    print('epoch {}'.format(epoch))
                    print('train_loss: {:.4f}'.format(train_loss))
                    # test_loss, _, _ = validate(model, test_loader, criterion, test_times, use_cuda)
                    # if test_loss < best_loss:
                    #     best_epoch = epoch
                    #     best_loss = test_loss
                    # print('test_loss: {:.4f}'.format(test_loss))
                    test_loss = np.nan
                    checkpoint_state = {
                        'epoch': epoch + 1,
                        'model_name': model_name,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'dataset_name': dataset_name
                    }
                    epoch2loss_train[epoch] = train_loss
                    epoch2loss_test[epoch] = test_loss
                    torch.save(checkpoint_state, "./save_model/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth.tar".format(dataset_name, model_name, batch_size, weight_decay, learning_rate, fix_noiselevel, init_noiselevel, dropout_prob, best_epochs, repeat_time))
                line = (dataset_name, model_name, batch_size, weight_decay, learning_rate, fix_noiselevel, init_noiselevel, dropout_prob, epoch2loss_train, epoch2loss_test, best_epoch, best_loss, test_loss)
                logger.info(line)
# %%
