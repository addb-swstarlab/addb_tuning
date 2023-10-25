import os
import pandas as pd
import numpy as np
import argparse

import baxus
from baxus.util.behaviors import BaxusBehavior
from baxus.util.behaviors.gp_configuration import GPBehaviour
from baxus.baxus import BAxUS
from baxus.util.parsing import embedding_type_mapper, acquisition_function_mapper, mle_optimization_mapper

from envs.spark import SparkParameters
from models.utils import get_logger, get_foldername, get_filename
from models.bench import SparkBench
from models.configs import baxus_params as bp
import models.configs as cfg

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bo', type=str, default='baxus')
parser.add_argument('-q', '--query', type=int, default=1, help='Define a number of tpc-h query to test')

logger = get_logger()
os.system('clear')
opt = parser.parse_args()
logger.info("## ARGUMENT INFORMATION ##")
for _ in vars(opt):
    logger.info(f"{_}: {vars(opt)[_]}")

os.makedirs(cfg.INCUMBENTS_RESULTS_PATH, exist_ok=True)

def main():
    bin_sizing_method = embedding_type_mapper[bp['embedding_type']]
    acquisition_function = acquisition_function_mapper[bp['acquisition_function']]
    mle_optimization_method = mle_optimization_mapper[bp['mle_optimization']]

    behavior = BaxusBehavior(n_new_bins=bp['new_bins_on_split'],
                            initial_base_length=bp['l_init'],
                            min_base_length=bp['l_min'],
                            max_base_length=bp['l_max'],
                            acquisition_function=acquisition_function,
                            embedding_type=bin_sizing_method,
                            adjust_initial_target_dim=bp['adjust_initial_target_dim'],
                            noise=bp['noise_std'],
                            budget_until_input_dim=bp['budget_until_input_dim']
                            )

    gp_behaviour = GPBehaviour(
        mll_estimation=mle_optimization_method,
        n_initial_samples=bp['multistart_samples'],
        n_best_on_lhs_selection=bp['multistart_after_samples'],
        n_mle_training_steps=bp['mle_training_steps'],
    )

    sp = SparkParameters()

    input_dim = len(sp) # args.input_dim
    ub = sp.ub
    lb = sp.lb

    f = SparkBench(dim=input_dim, ub=ub, lb=lb, sp=sp, bench_type=opt.query, history_dir=get_foldername('history'))
    
    optim = BAxUS(f=f,
                n_init=bp['n_init'],
                max_evals=bp['max_evals'],
                target_dim=bp['target_dim'],
                behavior=behavior,
                gp_behaviour=gp_behaviour,
                run_dir=get_foldername(os.path.join('bo-results', opt.bo))
                )

    optim.optimize()
    
    incumbent_configs, incumbent_results = optim.optimization_results_incumbent()
    np.save(get_filename(cfg.INCUMBENTS_RESULTS_PATH,'tuned_configs', '.npy'), incumbent_configs)
    np.save(get_filename(cfg.INCUMBENTS_RESULTS_PATH,'tuned_results', '.npy'), incumbent_results)
    
    # np.save(os.path.join(cfg.INCUMBENTS_RESULTS_PATH, 'tuned_configs.npy'), incumbent_configs)
    # np.save(os.path.join(cfg.INCUMBENTS_RESULTS_PATH, 'tuned_results.npy'), incumbent_results)
    

if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    else:
        logger.handlers.clear()
