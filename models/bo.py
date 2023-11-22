import pandas as pd
import numpy as np
import torch, time
import logging, os

import models.configs as cfg
from models.cgp import CustomGP
from models.bench import SparkBench
from models.utils import get_filename

from botorch.fit import fit_gpytorch_model
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

from gpytorch.mlls import ExactMarginalLogLikelihood


class BO_Tuner:
    def __init__(self, sb: SparkBench, trg_wk: int, history_data_path: str):
        self.sb = sb # Spark Benchmark class
        self.trg_wk = trg_wk
        self.history_data_path = history_data_path
        self.best_observed_configs_all = []
        self.best_observed_res_all = []
        
        self._set_bounds()
    
    def _set_bounds(self):
        lower = torch.Tensor(self.sb.lb)
        upper = torch.Tensor(self.sb.ub)
        
        self.bounds = torch.stack([lower, upper])
        self.bounds = torch.concat([self.bounds, torch.ones(2,1)*self.trg_wk], dim=1) # add condition range
        
        
    def _get_best_observed(self, data:pd.DataFrame) -> (float, pd.Series):
        '''
            data : raw data (pd.read_csv(path)) --> [knobs; workload_type; RATE]
        '''
        configs = data.iloc[:,:-2]
        res = - data[['res']]
        wks = data[cfg.QUERY_FEATURE_NAMES]
        
        trg_equal_wks_idx = wks.eq(self.sb.get_workload_feature(self.trg_wk)).all(axis=1)
        if sum(trg_equal_wks_idx) == 0:
            logging.info("There is no history data, replacing similar workload data.")
            ## TODO: instead of replacing, running default? setting to get results
            s =  wks.eq(self.sb.get_workload_feature(self.trg_wk)).sum(axis=1)
            max_idx = res[s==s.max()].idxmax().item()
        else:
            max_idx = res[wks.eq(self.sb.get_workload_feature(self.trg_wk)).all(axis=1)].idxmax().item()
        
        best_observed_res = res.loc[max_idx].item()
        best_observed_config = configs.loc[max_idx]
        
        return best_observed_config, best_observed_res
    
    def load_history_data(self):
        _data = pd.read_csv(self.history_data_path, index_col=0)
        data = _data.copy()
        train_x = data.iloc[:,:-2]
        ##### Preprocess data removing unit and converting boolean into int #####
        train_x.replace(False, 0, inplace=True)
        train_x.replace(True, 1, inplace=True)
        train_x.replace(to_replace=r'[kgm]',value='', regex=True, inplace=True)
        train_x = train_x.astype(float)
        #########################################################################
        train_wk = data[cfg.QUERY_FEATURE_NAMES]
        train_y = - data[['res']]
        
        train_x_knobs = torch.tensor(train_x.values)#.cuda()
        train_x_wks = torch.tensor(train_wk.values)#.cuda()
        train_y = torch.tensor(train_y.values).squeeze()#.cuda()
        
        best_observed_config, best_observed_res = self._get_best_observed(data)
        
        return train_x_knobs, train_x_wks, train_y, best_observed_config, best_observed_res
    
    def save_history_data(self,x_knobs:torch.Tensor, x_wks:torch.Tensor, y:torch.Tensor):
        y = y.unsqueeze(1) if y.dim() != 2 else y
        
        knobs = x_knobs.cpu().detach().numpy()
        wks = x_wks.cpu().detach().numpy()
        res = - y.cpu().detach().numpy()
        
        cols = pd.read_csv(self.history_data_path, index_col=0).columns
        datas = np.concatenate([knobs, wks, res], axis=1)

        saved_data = pd.DataFrame(data=datas, columns=cols)
        
        if os.path.exists(self.history_data_path):
            history_saved_path = get_filename(cfg.SAVE_HISTORY_FOLDER_PATH,'history_feature_data','.csv')
            os.system(f'cp {self.history_data_path} {os.path.join(cfg.SAVE_HISTORY_FOLDER_PATH, history_saved_path)}')
            logging.info(f"## Saved old history data to .. {os.path.join(cfg.SAVE_HISTORY_FOLDER_PATH, history_saved_path)}")
            
        saved_data.to_csv(self.history_data_path)
        
    def initialize_model(self, train_x1, train_x2, train_y, state_dict=None):
        model = CustomGP(train_x1, train_x2, train_y)#.cuda()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)#.cuda()
        
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model

    def optimize_acqf_and_get_observation(self, acq_func, bounds, trg_wk): # maximize
        qf = self.sb.get_workload_feature(trg_wk)
        fixed_features = {len(self.sb)+_: qf[_] for _ in range(len(qf))}
        candidate, acq_value = optimize_acqf(acq_function=acq_func,
                                            bounds=bounds,
                                            q=1,
                                            num_restarts=5,
                                            raw_samples=20,
                                            fixed_features={22:trg_wk} # fixed a workload feature
                                            )
        
        new_x = candidate.cpu().detach().numpy()
        new_x = new_x.squeeze()[:-1] # remove query information
        new_x[:-2] = np.round(new_x[:-2])
        new_x[-2:] = np.round(new_x[-2:], 1)
        
        new_obj = self.sb.benchmark(new_x)
        new_obj = torch.tensor(new_obj)#.cuda()
        
        return candidate, new_obj

    def optimize(self):
        train_x_knobs, train_x_wks, train_y, best_observed_config, best_observed_res_value = self.load_history_data()

        mll, model = self.initialize_model(train_x_knobs, train_x_wks, train_y)
        
        self.best_observed_configs_all.append(best_observed_config)
        self.best_observed_res_all.append(best_observed_res_value)
        
        t0 = time.monotonic()
        
        fit_gpytorch_model(mll)
        
        from botorch.sampling import SobolQMCNormalSampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=10)
        
        qEI = qExpectedImprovement(model=model,
                                   best_f=train_y.max(),
                                   sampler=qmc_sampler)
        
        new_x, new_res = self.optimize_acqf_and_get_observation(qEI, self.bounds, self.trg_wk)
        
        train_x_knobs = torch.cat([train_x_knobs, new_x[:, :-1]])
        train_x_wks = torch.cat([train_x_wks, new_x[:, -1].unsqueeze(0)])
        train_y = torch.cat([train_y, new_res.unsqueeze(0)])
        
        self.save_history_data(train_x_knobs, train_x_wks, train_y)
        
        _, _, _, best_observed_config, best_observed_res_value = self.load_history_data()
    
        # self.best_observed_configs_all.append(best_observed_config)
        # self.best_observed_res_all.append(best_observed_res_value)
        
        t1 = time.monotonic()
        
        logging.info(f"\nbest_value qEI = {best_observed_res_value:4.2f}, time = {t1-t0:4.2f}")
        
