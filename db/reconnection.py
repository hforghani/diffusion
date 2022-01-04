import functools
import os
import time

import pymongo

from db.managers import DBManager
from settings import MONGO_URL
from settings import logger

MAX_AUTO_RECONNECT_ATTEMPTS = 10


def reconnect(wait_t=0.5):
    # Check mongod is down. Try to start it if down.
    try:
        mongo_client = pymongo.MongoClient(MONGO_URL)
        mongo_client.server_info()
    except pymongo.errors.ServerSelectionTimeoutError:
        logger.info('mongod is down. starting mongod service ...')
        os.system('service mongod start')
        logger.info('mongod started')
        logger.warning("PyMongo auto-reconnecting... Waiting %.1f seconds.", wait_t)
        time.sleep(wait_t)


def rerun_auto_reconnect(mongo_op_func):
    """ Run the function until no AutoReconnect error is raised """

    @functools.wraps(mongo_op_func)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_AUTO_RECONNECT_ATTEMPTS):
            try:
                return mongo_op_func(*args, **kwargs)
            except pymongo.errors.AutoReconnect as e:
                if attempt == MAX_AUTO_RECONNECT_ATTEMPTS - 1:
                    raise
                wait_t = 0.5 * pow(2, attempt)  # exponential back off
                reconnect(wait_t)

    return wrapper
