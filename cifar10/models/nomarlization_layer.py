import torch
import torch.nn as nn

class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        
        return input.sub(self.mean).div(self.std)


class noise_Normalize_layer(nn.Module):
    
    def __init__(self, mean, std, input_noise=False):
        super(noise_Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.input_noise = input_noise
        self.alpha_i = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        
    def forward(self, input):
        output = input.sub(self.mean).div(self.std)
        
        input_std = output.std().item()
        input_noise = output.clone().normal_(0, input_std)
        
        return output + input_noise*self.alpha_i*self.input_noise

# 下面这段暂时没用上
from scipy.stats import levy_stable
import numpy as np
def estimate_all_params(X, beta=None):
    import levy
#     X = (X - X.mean())/X.std()

    params = dict()
#     params["mu"], params['sigma'] = 0., 1.
    if beta is not None: 
        params["beta"] = beta
    
    params, neglog_density = levy.fit_levy(X)
    p = params.__dict__
    r = dict(zip(p["pnames"], p["_x"]))
    r["log_density"] = -neglog_density
    return [
        np.float32(r['alpha']),
        np.float32(r['beta']),
        np.float32(r['sigma']),
        np.float32(r['mu'])
    ]

class noise2_Normalize_layer(nn.Module):
    
    def __init__(self, mean, std, input_noise=False):
        super(noise2_Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.input_noise = input_noise
        self.alpha_i = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        
    def forward(self, input):
        output = input.sub(self.mean).div(self.std)
        
        alpha, beta, gamma, mu = estimate_all_params(output.detach())
        input_noise = levy_stable.rvs(alpha=alpha,beta=0, loc=0, scale=gamma, size=output.shape)
        input_noise = torch.Tensor(input_noise)
        
        return output + input_noise*self.alpha_i*self.input_noise
