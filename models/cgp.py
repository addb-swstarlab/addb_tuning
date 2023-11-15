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
