import os, logging
import subprocess
import numpy as np
from models.configs import *

class SparkBench:
    def __init__(self, sp, bench_type=None, history_dir=None):
        self.dim = len(sp)
        self.ub = sp.ub
        self.lb = sp.lb
        self.sp = sp # spark parameter infos
        self.bench_type = bench_type
    
    def benchmark(self, x:np.array) -> float:
        spark_conf_dict = self.sp.save_configuration_file(x)
        self.execute_spark_bench()
        try:
            self.res = - self.get_results()
        except IndexError:
            logging.error("Invalid Configurations, pass testing this configuration")
            self.res = - 10000
        logging.info("############################")
        logging.info(f"##### runtime: {-self.res:.2f} ######")
        logging.info("############################")
        
        return round(self.res, 2)
        
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
