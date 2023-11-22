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

HISTORY_DATA_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'history_feature_data.csv')
SAVE_HISTORY_FOLDER_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'history')

QUERY_FEATURE_TABLE_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'query_tables.csv')

def set_query_feature_name(cols):
    global QUERY_FEATURE_NAMES
    QUERY_FEATURE_NAMES = cols
