import logging
import os
from datetime import datetime

LOG_FORMAT = '[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s'
LOG_LEVEL = logging.DEBUG

logging.basicConfig(format=LOG_FORMAT)
file_handler = logging.FileHandler('log/testpredict-{}.log'.format(datetime.now().strftime('%Y%m%d-%H%M%S')), 'w',
                                   'utf-8')
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger = logging.getLogger()
logger.addHandler(file_handler)
logger.setLevel(LOG_LEVEL)

VERBOSITY = 2

BASEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')

THRESHOLDS = {
    'memm': (-0.01, 1),
    'mlnprac': (5, 10),
    'mlnalch': (-0.01, 0.5),
    'aslt': (-0.01, 1),
    'avg': (-0.001, 0.1),
}
