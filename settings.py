import logging
import os

LOG_FORMAT = '[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
LOG_LEVEL = logging.DEBUG

VERBOSITY = 2

BASEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
