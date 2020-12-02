# -*- coding: utf-8 -*-
import logging
import time
from datetime import timedelta, datetime
import pytz

from settings import logger, DEBUG_LEVELV_NUM

DT_FORMAT = '%Y-%m-%d %H:%M:%S'


def localize(dt):
    return pytz.timezone('Asia/Tehran').localize(dt)


def str_to_datetime(datetime_str, dt_format=None):
    if not dt_format:
        dt_format = DT_FORMAT
    dt = datetime.strptime(datetime_str, dt_format)
    return localize(dt)


levels = {'debugv': DEBUG_LEVELV_NUM,
          'debug': logging.DEBUG,
          'info': logging.INFO}


def time_report(t, unit):
    if unit == 's':
        return f'{t} s'
    elif unit == 'm':
        return f'{round(t / 60)} m'
    elif unit == 'h':
        return f'{round(t / (60 * 60))} h'
    elif unit == 'd':
        return f'{round(t / (60 * 60 * 24))} days'
    else:
        raise ValueError(f'invalid unit "{unit}"')


def time_measure(level='info', unit='s'):
    def decorator(func):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            res = func(*args, **kwargs)
            t = time.time() - t0
            time_expr = time_report(t, unit)
            logger.log(levels[level], f'function "{func.__name__}" executed in {time_expr}')
            return res

        return wrapper

    return decorator


class Timer:
    def __init__(self, name, level='info', unit='s'):
        self.name = name
        self.level = level
        self.unit = unit

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time() - self.start
        time_expr = time_report(t, self.unit)
        logger.log(levels[self.level], f'the code "{self.name}" executed in {time_expr}')
