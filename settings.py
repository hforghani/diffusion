import logging
import os
from datetime import datetime
import pymongo
from local_settings import *

LOG_FORMAT = '[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s'
LOG_LEVEL = logging.INFO
logging.basicConfig(format=LOG_FORMAT)
file_handler = logging.FileHandler('{}/log/{}.log'.format(BASE_PATH, datetime.now().strftime('%Y%m%d-%H%M%S')), 'w',
                                   'utf-8')
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger = logging.getLogger()
logger.addHandler(file_handler)
logger.setLevel(LOG_LEVEL)

VERBOSITY = 2

BASEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongodb = mongo_client[DB_NAME]
