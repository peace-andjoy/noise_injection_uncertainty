# 好像不是最终版？
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

# %%
res = []
# dataset_names = ['bostonHousing', 'concrete', 'energy', 'kin8nm', 'power-plant', 'protein-tertiary-structure', 'wine-quality-red', 'yacht']
# model_names = ['FCN', 'noisyFCN', 'fixnoiseFCN', 'dropoutFCN']
dataset_names = ['bostonHousing', 'concrete', 'energy', 'kin8nm', 'power-plant', 'protein-tertiary-structure', 'wine-quality-red', 'yacht']
model_names = ['FCN', 'noisyFCN', 'fixnoiseFCN', 'dropoutFCN']

for dataset_name in dataset_names:
    for model_name in model_names:
        filelist = []
        for file in os.listdir("./save_model/"):
            if '{}_{}'.format(dataset_name, model_name) in file:
                filelist.append(file)
        if len(filelist) != 5:
            print('{}, {}, not 5!'.format(dataset_name, model_name))
            print(filelist)
            break
        loss_list = []
        sqrt_loss_list = []
        msll_list = []
        for file in filelist:
            dataset_name, model_name, batch_size, weight_decay, learning_rate, fix_noiselevel, init_noiselevel, dropout_prob, best_epochs, _  = file.split('_')
            batch_size, weight_decay, learning_rate, fix_noiselevel, init_noiselevel, dropout_prob, best_epochs = int(batch_size), float(weight_decay), float(learning_rate), float(fix_noiselevel), float(init_noiselevel), float(dropout_prob), int(best_epochs)
            checkpoint = os.path.join("./save_model/", file)
            X_train, y_train, X_test, y_test = load_uci_data_test(dataset_name)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            test_data = torch.utils.data.TensorDataset(X_test, y_test)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
            eval_path = os.path.join(os.getcwd(), 'evaluations')
            inverted_cv_fraction = 5
            # Create eval dir for dataset
            dataset_path = os.path.join(eval_path, dataset_name)
            if dataset_name == 'protein-tertiary-structure':
                hidden_sizes = [100, 100]
            else:
                hidden_sizes = [50, 50]
            use_cuda = torch.cuda.is_available()

            # Get dataset configuration
            feature_indices, target_indices = load_uci_info(dataset_name)
            input_size = len(feature_indices)
            output_size = len(target_indices)

            # %%

            def validate(model, val_loader, criterion, test_times=100):
                model.eval()
                if 'Dropout' in model._get_name():
                    recursive_module_iteration(model)
                losses = AverageMeter()
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
                        if use_cuda:
                            var_outputs = var_outputs.cpu()
                            avg_outputs = avg_outputs.cpu()
                        var_outputs = var_outputs.numpy().tolist()
                        var_outputs_all.extend(var_outputs)
                        avg_outputs = avg_outputs.numpy().tolist()
                        pred_all.extend(avg_outputs)
                return losses.avg, var_outputs_all, pred_all

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

            # %%
            model = def_model(model_name, input_size, hidden_sizes, output_size, init_noiselevel, fix_noiselevel, dropout_prob)
            criterion = nn.MSELoss()
            if use_cuda:
                model.cuda()
                criterion.cuda()

            checkpoint = torch.load(checkpoint)

            state_tmp = model.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)

            model.load_state_dict(state_tmp)
            print("=> loaded model '{}' (epoch {})".format(file, checkpoint['epoch']))

            test_loss, var_outputs_all, pred_all = validate(model, test_loader, criterion, test_times=100)
            # print(var_outputs_all)
            var_outputs_all = np.asarray(var_outputs_all)
            std_outputs_all = np.sqrt(var_outputs_all)
            std_outputs_all = std_outputs_all #/np.std(pred_all)/np.linalg.norm(std_outputs_all)
            print(std_outputs_all)
            msll = mean_standardized_log_loss(test_data.tensors[1].numpy().flatten(), pred_all, std_outputs_all)
            loss_list.append(test_loss)
            sqrt_loss_list.append(np.sqrt(test_loss))
            msll_list.append(msll)

        loss_mean = np.mean(loss_list)
        loss_std = np.std(loss_list)

        sqrt_loss_mean = np.mean(sqrt_loss_list)
        sqrt_loss_std = np.std(sqrt_loss_list)

        msll_mean = np.mean(msll_list)
        msll_std = np.std(msll_list)

        print(dataset_name, model_name, loss_mean, loss_std, sqrt_loss_mean, sqrt_loss_std, msll_mean, msll_std)

        res.append([dataset_name, model_name, loss_mean, loss_std, sqrt_loss_mean, sqrt_loss_std, msll_mean, msll_std])

# %%
res = pd.DataFrame(res, columns=['dataset_name', 'model_name', 'loss_mean', 'loss_std', 'sqrt_loss_mean', 'sqrt_loss_std', 'msll_mean', 'msll_std'])
print(res)
res.to_csv('predictive_res.csv', index=False)
