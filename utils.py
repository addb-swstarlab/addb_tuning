import datetime
import os, logging
import pandas as pd

def get_filename(PATH, head, tail):
    i = 0
    today = datetime.datetime.now()
    today = today.strftime('%Y%m%d')
    if not os.path.exists(os.path.join(PATH, today)):
        os.mkdir(os.path.join(PATH, today))
    name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    while os.path.exists(os.path.join(PATH, name)):
        i += 1
        name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    return name

def get_logger(log_path='./logs'):

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = logging.getLogger()
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('%(asctime)s[%(levelname)s] %(filename)s:%(lineno)s  %(message)s', date_format)
    name = get_filename(log_path, 'log', '.log')
    
    fileHandler = logging.FileHandler(os.path.join(log_path, name))
    streamHandler = logging.StreamHandler()
    
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    
    logger.setLevel(logging.INFO)
    logger.info('Writing logs at {}'.format(os.path.join(log_path, name)))
    return logger, os.path.join(log_path, name)

def save_results(addb, results):
    redis_bool_knobs = ['lazyfree-lazy-expire', 'lazyfree-lazy-server-del', 'lazyfree-lazy-eviction', 'activerehashing']
    spark_bool_knobs = ['spark.broadcast.compress', 'spark.rdd.compress', 'spark.shuffle.compress', 'spark.shuffle.spill.compress',
                        'spark.sql.codegen.aggregate.map.twolevel.enable', 'spark.sql.inMemoryColumnarStorage.compressed',
                        'spark.sql.inMemoryColumnarStorage.partitionPruning', 'spark.sql.join.preferSortMergeJoin',
                        'spark.sql.sort.enableRadixSort']
    
    pd_redis = pd.DataFrame(data=[results[:addb.redis_len].astype(int)], columns=addb.redis_knobs_columns)
    pd_rocksdb = pd.DataFrame(data=[results[addb.redis_len:addb.redis_len+addb.rocksdb_len].astype(int)], columns=addb.rocksdb_knobs_columns)
    pd_spark = pd.DataFrame(data=[results[-addb.spark_len:].astype(int)], columns=addb.spark_knobs_columns)
    
    if not os.path.exists('results'):
        os.mkdir('results')
    
    