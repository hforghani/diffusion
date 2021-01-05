import functools
import time

import pymongo

from settings import logger

MAX_AUTO_RECONNECT_ATTEMPTS = 5


def graceful_auto_reconnect(mongo_op_func):
    """Gracefully handle a reconnection event."""

    @functools.wraps(mongo_op_func)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_AUTO_RECONNECT_ATTEMPTS):
            try:
                return mongo_op_func(*args, **kwargs)
            except pymongo.errors.AutoReconnect as e:
                wait_t = 0.5 * pow(2, attempt)  # exponential back off
                logger.warning("PyMongo auto-reconnecting... %s. Waiting %.1f seconds.", str(e), wait_t)
                time.sleep(wait_t)

    return wrapper
