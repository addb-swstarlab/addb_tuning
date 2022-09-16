import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Redis():
    def __init__(self):
        self.KNOB_PATH = 'data/configs/Redis'
        self.RES_PATH = 'data/results/Redis_results.csv'
        self.size = len(os.listdir(self.KNOB_PATH))
        self.knobs = self.get_knobs()
        self.results = self.get_results()
        self.results_len = self.results.shape[1]
        self.split_data()
        self.scale_data()
        
    def get_knobs(self):
        knobs = pd.DataFrame()
        for idx in range(self.size):
            f = open(os.path.join(self.KNOB_PATH, f'config{idx+1}.conf'), 'r')
            conf = f.readlines()[-7:]
            f.close()
            d_conf = {}
            for i in  range(len(conf)):
                col, d = conf[i].split()
                if d == 'yes':
                    d = 1
                elif d == 'no':
                    d = 0
                d_conf[col] = d
            p_conf = pd.DataFrame([d_conf])
            knobs = pd.concat([knobs, p_conf])
        knobs = knobs.drop(columns=['auto-aof-rewrite-min-size'])
        self.knobs_columns = knobs.columns
        return knobs
      
    def get_results(self):
        res = pd.read_csv(self.RES_PATH)
        res = res[['Totals_Ops/sec', 'Totals_p99_Latency']]
        return res
    
    def split_data(self):
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(self.knobs, self.results, test_size=0.2, random_state=1004)

    def scale_data(self):
        self.scaler_X = MinMaxScaler().fit(self.X_tr)
        self.scaler_y = MinMaxScaler().fit(self.y_tr)
        
        self.scaled_X_tr = torch.Tensor(self.scaler_X.transform(self.X_tr)).cuda()
        self.scaled_X_te = torch.Tensor(self.scaler_X.transform(self.X_te)).cuda()
        self.scaled_y_tr = torch.Tensor(self.scaler_y.transform(self.y_tr)).cuda()
        self.scaled_y_te = torch.Tensor(self.scaler_y.transform(self.y_te)).cuda()
        
class RocksDB():
    def __init__(self):
        self.KNOB_PATH = 'data/configs/RocksDB'
        self.RES_PATH = 'data/results/RocksDB_results.csv'
        self.size = len(os.listdir(self.KNOB_PATH))
        self.knobs = self.get_knobs()
        self.results = self.get_results()
        self.results_len = self.results.shape[1]
        self.split_data()
        self.scale_data()
        
    def get_knobs(self):
        knobs = pd.DataFrame()
        compression_type = ["snappy", "none", "lz4", "zlib"]
        cache_index_and_filter_blocks = ["false", "true"]
        
        for idx in range(self.size):
            f = open(os.path.join(self.KNOB_PATH, f'config{idx+1}.cnf'), 'r')
            conf = f.readlines()[1:-1]
            f.close()
            d_conf = {}
            cmp_type=0
            for l in conf:
                col, _, d = l.split()
                if d in compression_type:
                    if d == 'none':
                        cmp_type=1
                    d = compression_type.index(d)
                elif d in cache_index_and_filter_blocks:
                    d = cache_index_and_filter_blocks.index(d)
                if col == "compression_ratio" and cmp_type:
                    d = 1                
                d_conf[col] = d
            p_conf = pd.DataFrame([d_conf])
            knobs = pd.concat([knobs, p_conf])
        knobs = knobs.drop(columns=['compaction_pri', 'compaction_style', 'compression_type', 'cache_index_and_filter_blocks',
                                    'memtable_bloom_size_ratio', 'compression_ratio'])
        self.knobs_columns = knobs.columns
        return knobs
    
    def get_results(self):
        res = pd.read_csv(self.RES_PATH, index_col=0)
        return res
    
    def split_data(self):
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(self.knobs, self.results, test_size=0.2, random_state=1004)

    def scale_data(self):
        self.scaler_X = MinMaxScaler().fit(self.X_tr)
        self.scaler_y = MinMaxScaler().fit(self.y_tr)
        
        self.scaled_X_tr = torch.Tensor(self.scaler_X.transform(self.X_tr)).cuda()
        self.scaled_X_te = torch.Tensor(self.scaler_X.transform(self.X_te)).cuda()
        self.scaled_y_tr = torch.Tensor(self.scaler_y.transform(self.y_tr)).cuda()
        self.scaled_y_te = torch.Tensor(self.scaler_y.transform(self.y_te)).cuda()

class Spark():
    def __init__(self):
        self.KNOB_PATH = 'data/configs/Spark'
        self.RES_PATH = 'data/results/Spark_results.csv'
        self.size = len(os.listdir(self.KNOB_PATH))
        self.knobs = self.get_knobs()
        self.results = self.get_results()
        self.results_len = self.results.shape[1]
        self.split_data()
        self.scale_data()
        
    def get_knobs(self):
        knobs = pd.DataFrame()
        for idx in range(self.size):
            f = open(os.path.join(self.KNOB_PATH, f'addb_config{idx}.conf'), 'r')
            conf = f.readlines()[1:]
            f.close()
            d_conf = {}
            for i in  range(len(conf)):
                col, d = conf[i].split()
                if d == 'true':
                    d = 1
                elif d == 'false':
                    d = 0
                d_conf[col] = d
            p_conf = pd.DataFrame([d_conf])
            knobs = pd.concat([knobs, p_conf])
        self.knobs_columns = knobs.columns
        return knobs
    
    def get_results(self):
        res = pd.read_csv(self.RES_PATH).T
        return res

    def split_data(self):
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(self.knobs, self.results, test_size=0.2, random_state=1004)

    def scale_data(self):
        self.scaler_X = MinMaxScaler().fit(self.X_tr)
        self.scaler_y = MinMaxScaler().fit(self.y_tr)
        
        self.scaled_X_tr = torch.Tensor(self.scaler_X.transform(self.X_tr)).cuda()
        self.scaled_X_te = torch.Tensor(self.scaler_X.transform(self.X_te)).cuda()
        self.scaled_y_tr = torch.Tensor(self.scaler_y.transform(self.y_tr)).cuda()
        self.scaled_y_te = torch.Tensor(self.scaler_y.transform(self.y_te)).cuda()
        
        
class ADDB():
    def __init__(self):
        self.KNOB_PATH = 'data/configs/ADDB'
        self.RES_PATH = 'data/results/ADDB_results.csv'
        self.size = 100
        self.redis_knobs = self.redis_get_knobs()
        self.rocksdb_knobs = self.rocksdb_get_knobs()
        self.spark_knobs = self.spark_get_knobs()
        self.redis_len = len(self.redis_knobs_columns)
        self.rocksdb_len = len(self.rocksdb_knobs_columns)
        self.spark_len = len(self.spark_knobs_columns)
        self.results = self.get_results()
        self.results_len = self.results.shape[1]
        self.split_data()
        self.scale_data()
        self.get_range()
        
    def redis_get_knobs(self):
        knobs = pd.DataFrame()
        for idx in range(self.size):
            f = open(os.path.join(self.KNOB_PATH, 'redis', f'addb_config{idx}.conf'), 'r')
            conf = f.readlines()[1:]
            f.close()
            d_conf = {}
            for i in  range(len(conf)):
                col, d = conf[i].split()
                if d == 'yes':
                    d = 1
                elif d == 'no':
                    d = 0
                d_conf[col] = d
            p_conf = pd.DataFrame([d_conf])
            knobs = pd.concat([knobs, p_conf])
        self.redis_knobs_columns = knobs.columns
        return knobs
        
    def rocksdb_get_knobs(self):
        knobs = pd.DataFrame()
        compression_type = ["snappy", "none", "lz4", "zlib"]
        cache_index_and_filter_blocks = ["false", "true"]
        
        for idx in range(self.size):
            f = open(os.path.join(self.KNOB_PATH, 'rocksdb', f'addb_config{idx}.conf'), 'r')
            conf = f.readlines()[1:]
            f.close()
            d_conf = {}
            cmp_type=0
            for l in conf:
                col, d = l.split()
                if d in compression_type:
                    if d == 'none':
                        cmp_type=1
                    d = compression_type.index(d)
                elif d in cache_index_and_filter_blocks:
                    d = cache_index_and_filter_blocks.index(d)
                if col == "compression_ratio" and cmp_type:
                    d = 1                
                d_conf[col] = d
            p_conf = pd.DataFrame([d_conf])
            knobs = pd.concat([knobs, p_conf])
        self.rocksdb_knobs_columns = knobs.columns
        return knobs
        
    def spark_get_knobs(self):
        knobs = pd.DataFrame()
        for idx in range(self.size):
            f = open(os.path.join(self.KNOB_PATH, 'spark', f'addb_config{idx}.conf'), 'r')
            conf = f.readlines()[1:]
            f.close()
            d_conf = {}
            for i in  range(len(conf)):
                col, d = conf[i].split()
                if d == 'true':
                    d = 1
                elif d == 'false':
                    d = 0
                d_conf[col] = d
            p_conf = pd.DataFrame([d_conf])
            knobs = pd.concat([knobs, p_conf])
        self.spark_knobs_columns = knobs.columns
        return knobs
    
    def get_results(self):
        res = pd.read_csv(self.RES_PATH).T
        return res

    def split_data(self):
        self.redis_tr, self.redis_te, self.rocksdb_tr, self.rocksdb_te, self.spark_tr, self.spark_te, self.y_tr, self.y_te = \
            train_test_split(self.redis_knobs, self.rocksdb_knobs, self.spark_knobs, self.results, test_size=0.2, random_state=1004)

    def scale_data(self):
        self.scaler_redis = MinMaxScaler().fit(self.redis_tr)
        self.scaler_rocksdb = MinMaxScaler().fit(self.rocksdb_tr)
        self.scaler_spark = MinMaxScaler().fit(self.spark_tr)
        self.scaler_y = MinMaxScaler().fit(self.y_tr)
        
        self.scaled_redis_tr = torch.Tensor(self.scaler_redis.transform(self.redis_tr)).cuda()
        self.scaled_redis_te = torch.Tensor(self.scaler_redis.transform(self.redis_te)).cuda()
        self.scaled_rocksdb_tr = torch.Tensor(self.scaler_rocksdb.transform(self.rocksdb_tr)).cuda()
        self.scaled_rocksdb_te = torch.Tensor(self.scaler_rocksdb.transform(self.rocksdb_te)).cuda()
        self.scaled_spark_tr = torch.Tensor(self.scaler_spark.transform(self.spark_tr)).cuda()
        self.scaled_spark_te = torch.Tensor(self.scaler_spark.transform(self.spark_te)).cuda()
        self.scaled_y_tr = torch.Tensor(self.scaler_y.transform(self.y_tr)).cuda()
        self.scaled_y_te = torch.Tensor(self.scaler_y.transform(self.y_te)).cuda()
        
    def get_range(self):
        self.addb_lower_boundary = np.concatenate((np.array(self.redis_knobs.astype(float).min()), 
                                                   np.array(self.rocksdb_knobs.astype(float).min()),
                                                   np.array(self.spark_knobs.astype(float).min())))
        self.addb_upper_boundary = np.concatenate((np.array(self.redis_knobs.astype(float).max()), 
                                                   np.array(self.rocksdb_knobs.astype(float).max()),
                                                   np.array(self.spark_knobs.astype(float).max())))