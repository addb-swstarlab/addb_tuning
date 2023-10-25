import datetime
import os, logging

def get_filename(PATH, head, tail):
    i = 0
    today = datetime.datetime.now()
    today = today.strftime('%Y%m%d')
    os.makedirs(os.path.join(PATH, today), exist_ok=True)
    # if not os.path.exists(os.path.join(PATH, today)):
    #     os.mkdir(os.path.join(PATH, today))
    name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    while os.path.exists(os.path.join(PATH, name)):
        i += 1
        name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    return name

def get_foldername(PATH):
    i = 0
    today = datetime.datetime.now()
    today = today.strftime('%Y%m%d')
    os.makedirs(PATH, exist_ok=True)
    # if not os.path.exists(PATH):
    #     os.mkdir(PATH)
    folder_name = os.path.join(PATH, today + '-%02d'%i)
    while os.path.exists(folder_name):
        i += 1
        folder_name = os.path.join(PATH, today + '-%02d'%i)
    return folder_name
        
        
def get_logger(log_path='./logs'):
    os.makedirs(log_path, exist_ok=True)

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
    # return logger, os.path.join(log_path, name)
    return logger
