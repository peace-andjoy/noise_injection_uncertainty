import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FullyConnectedNN, self).__init__()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        
        # Create fully connected layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Output layer without activation for regression problems
        return x

# noisy_FCNN
class noise_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, pni='elementwise', w_noise=True, init_noiselevel = 0.1):
        super(noise_Linear, self).__init__(in_features, out_features, bias)
        
        self.pni = pni
        if self.pni is 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([init_noiselevel]), requires_grad = True)
        elif self.pni is 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_features).view(-1,1)*init_noiselevel,
                                        requires_grad=True)
        elif self.pni is 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*init_noiselevel, requires_grad = True)
        
        self.w_noise = w_noise

    def forward(self, input):
        
        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)

        noise_weight = self.weight + self.alpha_w * noise * self.w_noise
        output = F.linear(input, noise_weight, self.bias)
        
        return output 

class NoisyFullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, init_noiselevel):
        super(NoisyFullyConnectedNN, self).__init__()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        
        # Create fully connected layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(noise_Linear(layer_sizes[i], layer_sizes[i+1], init_noiselevel=init_noiselevel))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Output layer without activation for regression problems
        return x
    
# fixnoise_FCNN
class fixnoise_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, pni='layerwise', w_noise=True, fix_noiselevel=0.1):
        super(fixnoise_Linear, self).__init__(in_features, out_features, bias)
        
        self.pni = pni
        if self.pni is 'layerwise':
            self.alphafix_w = nn.Parameter(torch.Tensor([fix_noiselevel]), requires_grad = True)
        elif self.pni is 'channelwise':
            self.alphafix_w = nn.Parameter(torch.ones(self.out_features).view(-1,1)*fix_noiselevel,
                                        requires_grad=True)
        elif self.pni is 'elementwise':
            self.alphafix_w = nn.Parameter(torch.ones(self.weight.size())*fix_noiselevel, requires_grad = True)
        
        self.w_noise = w_noise

    def forward(self, input):
        
        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)

        noise_weight = self.weight + self.alphafix_w * noise * self.w_noise
        output = F.linear(input, noise_weight, self.bias)
        
        return output 

class FixNoiseFullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, fix_noiselevel=0.1):
        super(FixNoiseFullyConnectedNN, self).__init__()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        
        # Create fully connected layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(fixnoise_Linear(layer_sizes[i], layer_sizes[i+1], fix_noiselevel=fix_noiselevel))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Output layer without activation for regression problems
        return x

# dropout_FCNN

class DropoutFullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.05):
        super(DropoutFullyConnectedNN, self).__init__()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Create fully connected layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.dropouts.append(nn.Dropout(dropout_prob))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer, dropout in zip(self.layers[:-1], self.dropouts):
            x = self.relu(layer(x))
            x = dropout(x)

        x = self.layers[-1](x)  # Output layer without activation for regression problems
        return x
