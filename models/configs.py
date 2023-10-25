import os
from models.gcp_info import GCP_SPARK_MASTER_ADDRESS, GCP_INSTANCE_PASSWD, GCP_SPARK_CONF_PATH

HOME_PATH = os.path.expanduser('~')
PROJECT_NAME = 'Tuner'

SPARK_CONF_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'envs/test_spark.conf')
# SPARK_CONF_TEMPLATE_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'data/spark.conf.template')

MASTER_ADDRESS = GCP_SPARK_MASTER_ADDRESS
MASTER_CONF_PATH = GCP_SPARK_CONF_PATH
MASTER_USER_PASSWD = GCP_INSTANCE_PASSWD

TPCH_RESULT_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'envs/result.txt')

INCUMBENTS_RESULTS_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'results')

baxus_params = {'embedding_type': 'baxus',
                'acquisition_function': 'ts',
                'mle_optimization': 'sample-and-choose-best',
                'target_dim': 4, # 10
                'n_init': 10, # 10
                'max_evals': 20, # 100
                'noise_std': 0,
                'new_bins_on_split': 2,
                'multistart_samples': 100,
                'mle_training_steps': 50,
                'multistart_after_samples': 10,
                'l_init': 0.8,
                'l_min': 0.5**7,
                'l_max': 1.6,
                'adjust_initial_target_dim': True,
                'budget_until_input_dim': 0
                }
