import os
import numpy as np
import argparse

from envs.spark import SparkParameters
from models.utils import get_logger, get_foldername, get_filename
from models.bench import SparkBench
from models.configs import baxus_params as bp
import models.configs as cfg
from models.bo import BO_Tuner

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query', type=int, default=1, help='Define a number of tpc-h query to test')
parser.add_argument('-s', '--sqlfile', type=str, help='Provide sql file path to be tuned')
parser.add_argument('-t', '--trials', type=int, default=1, help='Define a number of iteration to tune')
parser.add_argument('--tuning', action='store_true', help='If you want to skip collecting observations(historical data) and to do actual tuning, trigger this')

logger = get_logger()
os.system('clear')
opt = parser.parse_args()
assert (opt.sqlfile is not None and opt.query is None) or (opt.sqlfile is None and opt.query is not None), "Please enter one options, --sqlfile test.sql or --query 1"
logger.info("## ARGUMENT INFORMATION ##")
for _ in vars(opt):
    logger.info(f"{_}: {vars(opt)[_]}")

os.makedirs(cfg.INCUMBENTS_RESULTS_PATH, exist_ok=True)

def main():
    sp = SparkParameters()
    
    sb = SparkBench(sp=sp, bench_type=opt.query, history_dir=get_foldername('history'))
    
    bo_tuner = BO_Tuner(sb=sb,
                        trg_wk=opt.query,
                        history_data_path=cfg.HISTORY_DATA_PATH)

    if opt.tuning:
        # Tuning mode
        logger.info("##########TUNING MODE##########")        
        bo_tuner.recommend()
    else:
        # Collecting mode
        for trial in range(1, opt.trials):
            logger.info(f"\n####Trials {trial:>2} of {opt.trials} ####")
            
            bo_tuner.optimize()
    
        INC_C_PATH = os.path.join(cfg.INCUMBENTS_RESULTS_PATH, get_filename(cfg.INCUMBENTS_RESULTS_PATH,'tuned_configs', '.npy'))
        INC_R_PATH = os.path.join(cfg.INCUMBENTS_RESULTS_PATH, get_filename(cfg.INCUMBENTS_RESULTS_PATH,'tuned_results', '.npy'))
        np.save(INC_C_PATH, bo_tuner.best_observed_configs_all)
        np.save(INC_R_PATH, bo_tuner.best_observed_res_all)
        logger.info("save incumbent at..")
        logger.info(f"configurations : {INC_C_PATH}")
        logger.info(f"results        : {INC_R_PATH}")


if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    else:
        logger.handlers.clear()
