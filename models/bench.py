import os, logging
import subprocess
import pandas as pd
import numpy as np
import models.configs as cfg
from envs.spark import SparkParameters
from envs.query_featurization import QueryFeatureGenerator

class SparkBench:
    def __init__(self, sp:SparkParameters, bench_type:int=None, sql_path:str=None):
        self.dim = len(sp)
        self.ub = sp.ub
        self.lb = sp.lb
        self.sp = sp # spark parameter infos
        self.bench_type = bench_type
        self.sql_path = sql_path
        self.parameter_names = sp.parameter_names

        assert (self.bench_type is not None and self.sql_path is None) or (self.bench_type is None and self.sql_path is not None), "Please use only one option, bench_type or sql_path"

        self._set_query_features()

        self.target_workload_feature = self.get_workload_feature(self.bench_type, self.sql_path)

    def __len__(self):
        return self.dim

    def _set_query_features(self):
        self.q_features = pd.read_csv(cfg.QUERY_FEATURE_TABLE_PATH)
        cfg.set_query_feature_name(list(self.q_features.columns))
        self.q_len = len(self.q_features.columns)
        if self.sql_path is not None:
            self.q_feat_gen = QueryFeatureGenerator(file_path=self.sql_path,
                                                    positional_encoding=True
                                                    )
            self.q_num = len(self.q_feat_gen.sequence_feature)
            with open(self.sql_path, 'r') as file:
                sql_query = file.readlines()
            logging.info("#########################################")
            logging.info(f"requested sql: {sql_query[0]}")
            logging.info("#########################################")
            file.close()
            del sql_query
        else:
            self.q_num = 1 if self.bench_type != 15 else 3

    def get_workload_feature(self, bench_type:int=None, sql_path:str=None) -> list:
        if bench_type is None:
            bench_type = self.bench_type
        if sql_path is None:
            sql_path = self.sql_path
        assert (bench_type is not None and sql_path is None) or (bench_type is None and sql_path is not None), f"Please use only one option, bench_type({bench_type}) or sql_path({sql_path}).."
        
        if bench_type is not None:
            b_feature = self.q_features.iloc[int(bench_type)-1].values
            b_feature = np.round(b_feature, 1)
        else:
            b_feature = self.q_feat_gen.workload_feature.values[0]
            b_feature = np.round(b_feature, 1)
        
        return list(b_feature)
    
    def benchmark(self, x:np.array) -> float:
        spark_conf_dict = self.sp.save_configuration_file(x)
        self._execute_spark_bench()
        try:
            self.res = - self._get_results()
        except IndexError:
            logging.error("Invalid Configurations, pass testing this configuration")
            self.res = -10000
        
        if self.res == 0:
            self.res = -10000
        
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
        
    def get_results(self) -> float:
        f = open(cfg.TPCH_RESULT_PATH, 'r')
        result_log = f.readlines()
        f.close()
        
        times = []
        for l in result_log:
            if l.find('seconds') != -1 and l.find('selected') != -1:
                runtime = l.split(' ')[-2][1:]
                times.append(runtime)

         ## If there are errors in several queries
        if len(times) != self.q_num:            
            res = 10000
            logging.info("**************************************************")
            logging.info("bad results showed..")
            logging.info(result_log)
            logging.info("**************************************************")
        else:
            res = np.sum(times, dtype=float)
        
        logging.info(f"Recorded times: {times}")
        
        return res
