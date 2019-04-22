import logging
import os

LOG_FORMAT = '[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
LOG_LEVEL = logging.DEBUG

VERBOSITY = 2

BASEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')

THRESHOLDS = {
    'memm': (-0.01, 1),
    'mlnprac': (5, 10),
    'mlnalch': (-0.01, 0.5),
    'aslt': (-0.01, 1),
    'avg': (-0.001, 0.1),
}
