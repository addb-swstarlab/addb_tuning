import utils
import pandas as pd
import numpy as np

import torch
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from botorch.models.gpytorch import GPyTorchModel
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, MaternKernel, AdditiveKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood



class CustomGP(ExactGP, GPyTorchModel):
    _num_outputs=1
    def __init__(self, train_knobs, train_wks, train_y):
        train_x_full = torch.cat([train_knobs, train_wks], dim=-1)
        super(CustomGP, self).__init__(train_x_full, train_y, GaussianLikelihood())
        
        self.knobs_kernel = ScaleKernel(MaternKernel(active_dims=torch.arange(0, train_knobs.size(1)), nu=2.5))
        
        self.wks_kernel = ScaleKernel(MaternKernel(active_dims=torch.arange(train_knobs.size(1), 
                                                                            train_knobs.size(1)+train_wks.size(1)), 
                                                   nu=2.5))
        self.mean_module = ConstantMean()
        self.covar_module = AdditiveKernel(self.knobs_kernel, self.wks_kernel)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
def initialize_model(train_x1, train_x2, train_y, state_dict=None):
    model = CustomGP(train_x1, train_x2, train_y).cuda()
    mll = ExactMarginalLogLikelihood(model.likelihood, model).cuda()
    
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

def optimize_acqf_and_get_observation(acq_func, bounds, trg_wk): # maximize
    candidate, acq_value = optimize_acqf(acq_function=acq_func,
                                         bounds=bounds,
                                         q=1,
                                         num_restarts=5,
                                         raw_samples=20,
                                         fixed_features={22:trg_wk} # fixed a workload feature
                                        )
    
    new_x = candidate.cpu().detach().numpy()
    ## TODO:
    ## Benchmarking step
    new_obj = regr.predict(new_x)
    new_x = torch.tensor(new_x).cuda()
    new_obj = torch.tensor(new_obj).cuda()
    return new_x, new_obj

def get_best_observed(data:pd.DataFrame, trg_wk:int) -> (float, pd.Series):
    '''
        data : raw data (pd.read_csv(path)) --> [knobs; workload_type; RATE]
    '''
    configs = data.iloc[:,:-2]
    res = data[['res']]
    wks = data[['workload_type']]
    
    max_idx = res[wks.workload_type==trg_wk].idxmax().item()
    
    best_observed_res = res.loc[max_idx].item()
    best_observed_config = configs.loc[max_idx]
    
    return best_observed_config, best_observed_res

def load_history_data(trg_wk, path=utils.HISTORY_DATA_PATH):
    data = pd.read_csv(path, index_col=0)
    train_x = data.iloc[:,:-2]
    train_wk = data[['workload_type']]
    train_y = data[['res']]
    
    train_x_knobs = torch.tensor(train_x.values).cuda()
    train_x_wks = torch.tensor(train_wk.values).cuda()
    train_y = torch.tensor(train_y.values).squeeze().cuda()
    
    best_observed_config, best_observed_res = get_best_observed(data, trg_wk)
    
    return train_x_knobs, train_x_wks, train_y, best_observed_config, best_observed_res

def save_history_data(x_knobs:torch.Tensor, x_wks:torch.Tensor, y:torch.Tensor):
    y = y.unsqueeze(1) if y.dim() != 2 else y
    
    knobs = x_knobs.cpu().detach().numpy()
    wks = x_wks.cpu().detach().numpy()
    res = y.cpu().detach().numpy()
    
    cols = pd.read_csv(utils.HISTORY_DATA_PATH, index_col=0).columns
    datas = np.concatenate([knobs, wks, res], axis=1)

    saved_data = pd.DataFrame(data=datas, columns=cols)
    
    saved_data.to_csv(utils.HISTORY_DATA_PATH)
