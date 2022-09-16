import os
import argparse
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from utils import get_logger
from dbmsinfos import Redis, RocksDB, Spark, ADDB
from train import train_model
from train_addb import addb_train_model
from ga import ADDBProblem, genetic_algorithm

os.system('clear')

parser = argparse.ArgumentParser()
parser.add_argument('--dbms', type=str, choices=['ADDB', 'Redis', 'RocksDB', 'Spark'], default='ADDB', help='choose DBMS')
parser.add_argument('--lr', type=float, default=0.001, help='Define learning rate')
parser.add_argument('--epochs', type=int, default=50, help='Define train epochs')
parser.add_argument('--hidden_dim', type=int, default=32, help='Define model hidden size')
parser.add_argument('--batch_size', type=int, default=32, help='Define model batch size')
parser.add_argument('--redis_param', type=str, help='Define redis prediction model pt path')
parser.add_argument('--rocksdb_param', type=str, help='Define rocksdb prediction model pt path')
parser.add_argument('--spark_param', type=str, help='Define spark prediction model pt path')
parser.add_argument('--population', type=int, default=100, help='Define pop_size on genetic algorithm')

opt = parser.parse_args()

if not os.path.exists('logs'):
    os.mkdir('logs')
    
if not os.path.exists('model_save'):
    os.mkdir('model_save')

# if not os.path.exists(os.path.join('model_save', opt.dbms)):
#     os.mkdir(os.path.join('model_save', opt.dbms))

logger, log_dir = get_logger(os.path.join('./logs'))

logger.info("## model hyperparameter information ##")
for i in vars(opt):
    logger.info(f'{i}: {vars(opt)[i]}')

def main():
    if opt.dbms == 'ADDB':
        dbms = ADDB()
    elif opt.dbms == 'Redis':
        dbms = Redis()
    elif opt.dbms == 'RocksDB':
        dbms = RocksDB()
    elif opt.dbms == 'Spark':
        dbms = Spark()


    if opt.dbms == 'ADDB':
        best_model, outputs = addb_train_model(dbms, opt)
    else:
        best_model, outputs = train_model(dbms, opt)
    
    true = dbms.y_te.to_numpy()
    pred = np.round(dbms.scaler_y.inverse_transform(outputs.cpu().detach().numpy()), 2)
    
    for i in range(10):
        logger.info(f'predict rslt: {pred[i]}')
        logger.info(f'ground truth: {true[i]}\n')
    
    mse_res = mean_squared_error(true, pred)
    pcc_res = np.zeros(dbms.results_len)
    for idx in range(dbms.results_len):
        res, _ = pearsonr(true[:,idx], pred[:,idx])
        pcc_res[idx] = res
        
    logger.info('[PCC SCORE]')
    logger.info(f'AVERAGE PCC SCORE : {np.average(pcc_res):.4f}')
    logger.info('[MSE SCORE]')
    logger.info(f'AVERAGE MSE SCORE : {mse_res:.4f}')
    
    if opt.dbms == 'ADDB':
        problem = ADDBProblem(dbms, best_model)
        res = genetic_algorithm(problem=problem, pop_size=opt.population)
        

if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    finally:
        logger.handlers.clear()
