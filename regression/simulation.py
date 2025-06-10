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
from matplotlib import pyplot as plt
from tqdm import tqdm

# %%
num_sample = 20
np.random.seed(123)
x = np.random.uniform(-4, 4, num_sample)
eps = np.random.normal(0, 3, num_sample)
y = x**3 + eps

# %%
x = x[:, np.newaxis]
y = y[:, np.newaxis]

# %%
X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
train_data = torch.utils.data.TensorDataset(X, y)
batch_size = len(X)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)

# %%
# ['FCN', 'noisyFCN', 'fixnoiseFCN', 'dropoutFCN']
model_name = "noisyFCN"
input_size = X.shape[1]
hidden_sizes = [100]
output_size = 1
init_noiselevel, fix_noiselevel, dropout_prob = 0.0001, 0.01, 0.01

model = def_model(model_name, input_size, hidden_sizes, output_size, init_noiselevel, fix_noiselevel, dropout_prob)

learning_rate = 0.005
momentum = 0.9

optimizer_name = "SGD"
normal_param = [
    param for name, param in model.named_parameters()
    if (not 'alpha_' in name and not 'alphafix_' in name)
] # this is the parameters do not contain noise scale coefficient
alpha_param = [
    param for name, param in model.named_parameters()
    if 'alpha_' in name
]
optimizer = def_optm(optimizer_name, normal_param, alpha_param, learning_rate, momentum=momentum, weight_decay=0)

criterion = nn.MSELoss()
use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()
    criterion.cuda()

global_max_epochs = 40

train_losses = []
for epoch in tqdm(range(global_max_epochs)):
    train_loss = train(model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion, use_cuda=use_cuda)
    print('epoch: {}, train_loss: {:.4f}'.format(epoch, train_loss))
    train_losses.append(train_loss)

# %%
plt.plot(train_losses)
plt.show()

# %%
new_x = np.arange(-4, 4.001, 0.001)
new_x = new_x[:, np.newaxis]
new_X = torch.tensor(new_x, dtype=torch.float32)
new_X = new_X.cuda()
new_y = model(new_X)
plt.figure(figsize=[4, 4], dpi=300)
plt.plot(new_x[:, 0], new_y.cpu().detach().numpy())
plt.scatter(x, y, s=4)
plt.show()

# %%
