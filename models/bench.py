import os, logging
import subprocess
import pandas as pd
from models.configs import *
from baxus.benchmarks.benchmark_function import Benchmark

class SparkBench(Benchmark):
    def __init__(self, dim, ub, lb, sp, bench_type=None, history_dir=None):
        super().__init__(dim=dim, ub=ub, lb=lb, noise_std=0)
        self.sp = sp # spark parameter infos
        self.bench_type = bench_type
        self.history_dir = history_dir
        
        if self.history_dir is not None:
            os.makedirs(self.history_dir, exist_ok=True)
    
    def _save_history(self, spark_conf_dict:dict, res):
        spark_conf_pd = pd.DataFrame.from_dict(data=spark_conf_dict, orient='index')
        
        if os.path.exists(os.path.join(self.history_dir, 'configuration.csv')):
            entire_spark_conf_pd = pd.read_csv(os.path.join(self.history_dir, 'configuration.csv'), index_col=0)
            entire_spark_conf_pd = pd.concat([entire_spark_conf_pd, spark_conf_pd], axis=1)
            entire_spark_conf_pd.to_csv(os.path.join(self.history_dir, 'configuration.csv'))
        else:
            spark_conf_pd.to_csv(os.path.join(self.history_dir, 'configuration.csv'))
            
        with open(os.path.join(self.history_dir, 'res.txt'), 'a') as f:
            f.write(f'{res}\n')
    
    def __call__(self, x):
        spark_conf_dict = self.sp.save_configuration_file(x)
        self.execute_spark_bench()
        self.res = self.get_results()
        logging.info("############################")
        logging.info(f"##### runtime: {self.res:.2f} ######")
        logging.info("############################")
        
        if self.history_dir is not None:
            self._save_history(spark_conf_dict, self.res)
        
        return self.res
        
    def execute_spark_bench(self):
        # if self.bench_type is not defined (self.bench_type is None), AssertionError
        assert self.bench_type is not None, 'please define bench_type'
        
        # transport a generated configuration to the master server
        os.system(f'sshpass -p {MASTER_USER_PASSWD} scp {SPARK_CONF_PATH} {MASTER_ADDRESS}:{MASTER_CONF_PATH}')
        
        ### !!!! NOTE !!!! ###
        ## ADDB should finish inserting tpch100g and keep start addb_spark.. ##
        ## In this tuning process, first step is to stop addb_spark ## 
        # 1. Stop addb_spark 
        os.system(f'sshpass -p {MASTER_USER_PASSWD} ssh -T {MASTER_ADDRESS} < /home/jieun/Tuner/scripts/stop_spark.sh')
        # 2. Start addb_spark with the generated configuration in MASTER_CONF_PATH
        os.system(f'sshpass -p {MASTER_USER_PASSWD} ssh -T {MASTER_ADDRESS} < /home/jieun/Tuner/scripts/start_spark.sh')
        # 3. Run tpch query               
        _, output = subprocess.getstatusoutput(f'sshpass -p {MASTER_USER_PASSWD} ssh -T {MASTER_ADDRESS} < /home/jieun/Tuner/scripts/run_tpch_q{self.bench_type}.sh')
        
        f = open(TPCH_RESULT_PATH, 'w')
        f.writelines(output)
        f.close()
        
    def get_results(self):
        f = open(TPCH_RESULT_PATH, 'r')
        result_log = f.readlines()
        f.close()
        
        times = []
        for l in result_log:
            if l.find('seconds') != -1:
                runtime = l.split(' ')[-2][1:]
                times.append(runtime)

        res = times[-1] if self.bench_type!=15 else times[1]
        return float(res)
