import os
from datetime import datetime
from local_settings import *  # Do not remove!
from local_params import *  # Do not remove!

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')

LOG_FORMAT = '[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] [%(processName)s] %(message)s'
logging.basicConfig(format=LOG_FORMAT)
file_handler = logging.FileHandler('{}/log/{}.log'.format(BASE_PATH, datetime.now().strftime('%Y%m%d-%H%M%S')), 'w',
                                   'utf-8')
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger = logging.getLogger('diffusion')
logger.addHandler(file_handler)
logger.setLevel(LOG_LEVEL)
