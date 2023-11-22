import os
import pandas as pd
import numpy as np
import argparse
from pyDOE import *
from scipy.stats.distributions import uniform

from envs.spark import SparkParameters
from models.bench import SparkBench
from models.utils import get_logger, get_foldername, get_filename

os.system('clear')

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sql', type=str, help="Define a path of a sql file")
parser.add_argument('-q', '--query', type=int, help='Define a number of tpc-h query')
parser.add_argument('--size', type=int, default=2, help="Define how many samples are collected")

opt = parser.parse_args()
assert (opt.sql is not None and opt.query is None) or (opt.sql is None and opt.query is not None), "Please enter one options, --sql test.sql or --query 1"
assert opt.sql is None, "TODO: applying to execute benchmark using sql files"

logger = get_logger()
logger.info("## ARGUMENT INFORMATION ##")
for _ in vars(opt):
    logger.info(f"{_}: {vars(opt)[_]}")

def get_LHS_random_samples(sp:SparkParameters, size:int)->np.ndarray:
    nfeats = len(sp)
    samples = lhs(nfeats, samples=size, criterion="maximin")

    maxvals = np.array(sp.ub)
    minvals = np.array(sp.lb)
    scales = maxvals - minvals

    for fidx in range(nfeats):
        samples[:, fidx] = uniform(loc=minvals[fidx], scale=scales[fidx]).ppf(samples[:,fidx])
    
    return samples

def main():
    sp = SparkParameters()
    sb = SparkBench(sp=sp, bench_type=opt.query, sql_path=opt.sql)
    cols = sb.parameter_names + list(sb.q_features.columns) + ['res']
    
    samples = get_LHS_random_samples(sp=sp, size=opt.size)
    
    history = []
    for sample in samples:
        res = sb.benchmark(sample)
        wf = sb.get_workload_feature()
        sample[:-2] = np.round(sample[:-2])
        sample[-2:] = np.round(sample[-2:], 1)
        hist = np.concatenate([sample, wf, [-res]])
        history.append(hist)
    
    pd_hist = pd.DataFrame(data=history, columns=cols)
    save_path = get_filename('observations',f'history_q{opt.query:02}', '.csv')
    pd_hist.to_csv(os.path.join('observations', save_path), index=None)

if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    else:
        logger.handlers.clear()
