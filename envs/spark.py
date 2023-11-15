import logging
from models.configs import SPARK_CONF_PATH

# Parameters based on Spark version 2.0.2
class SparkParameters():
    def __init__(self):
        self.boolean_parameters = {'spark.broadcast.compress': [['true', 'false'], 'true'],
                                  'spark.memory.offHeap.enabled': [['true', 'false'], 'true'],
                                  'spark.rdd.compress': [['true', 'false'], 'true'],
                                  'spark.shuffle.compress': [['true', 'false'], 'true'],
                                  'spark.shuffle.spill.compress': [['true', 'false'], 'true']
                                  }

        self.unit_mb_parameters = {'spark.broadcast.blockSize': [[1, 16], 4],
                                   'spark.kryoserializer.buffer.max': [[8, 128], 64],
                                   'spark.memory.offHeap.size': [[0, 49152], 0],
                                   'spark.reducer.maxSizeInFlight': [[24, 144], 48]
                                   }

        self.unit_kb_parameters = {'spark.kryoserializer.buffer': [[2, 128], 64],
                                   'spark.shuffle.file.buffer': [[16, 96], 32]
                                   }

        self.numerical_parameters = {'spark.default.parallelism':[[300, 1000], 300],
                                     'spark.locality.wait': [[1, 6], 3],
                                     'spark.scheduler.revive.interval': [[1, 5], 1],
                                     'spark.shuffle.io.numConnectionsPerPeer': [[1, 5], 1],
                                     'spark.shuffle.sort.bypassMergeThreshold': [[100, 1000], 200],
                                     'spark.speculation.interval': [[10, 1000], 100]
                                     }

        self.continous_parameters = {'spark.memory.fraction': [[0.1, 0.9], 0.6],
                                     'spark.memory.storageFraction': [[0.5, 0.9], 0.5]
                                     }
        
        self._get_combined_parameters()
        self._get_min_max_array()
        
        self.len_boolean = len(self.boolean_parameters)
        self.len_unit_kb = len(self.unit_kb_parameters)
        self.len_unit_mb = len(self.unit_mb_parameters)
        self.len_numerical = len(self.numerical_parameters)
        self.len_continuous = len(self.continous_parameters)
        
        self.parameter_names = list(self.all_parameters.keys())
        
    def __len__(self):
        return len(self.all_parameters)
    
    def _get_combined_parameters(self):
        self.all_parameters = self.boolean_parameters.copy()
        self.all_parameters.update(self.unit_kb_parameters)
        self.all_parameters.update(self.unit_mb_parameters)
        self.all_parameters.update(self.numerical_parameters)
        self.all_parameters.update(self.continous_parameters)
        
    def _get_min_max_array(self):
        self.ub = list()
        self.lb = list()
        for k, v in self.all_parameters.items():
            if k in self.boolean_parameters:
                self.lb.append(0)
                self.ub.append(1)
            else:
                self.lb.append(v[0][0])
                self.ub.append(v[0][1])
    
    def save_configuration_file(self, values):
        conf_file = open(SPARK_CONF_PATH, 'w')
        spark_conf_dict = {}
        # conf_file.writelines(self.conf_template)
        logging.info('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        for i, p in enumerate(self.parameter_names):
            if p in self.boolean_parameters:
                v = 'true' if round(values[i])==1 else 'false'
            elif p in self.continous_parameters:
                v = round(values[i], 1)
            elif p in self.unit_gb_parameters:
                v = str(round(values[i]))+'g'
            elif p in self.unit_kb_parameters:
                v = str(round(values[i]))+'k'
            elif p in self.unit_mb_parameters:
                v = str(round(values[i]))+'m'
            else:
                v = round(values[i])
            
            conf_file.writelines(f'{p}={v}\n')
            spark_conf_dict[p] = v
            logging.info(f'{p}={v}')
        logging.info('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        conf_file.close()
        return spark_conf_dict
