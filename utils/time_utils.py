# -*- coding: utf-8 -*-
import logging
import time
from datetime import timedelta, datetime
from enum import Enum

import pytz

from log_levels import DEBUG_LEVELV_NUM
from settings import logger

DT_FORMAT = '%Y-%m-%d %H:%M:%S'


class TimeUnit(Enum):
    SECONDS = 0
    MINUTES = 1
    HOURS = 2


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


def select_unit(t):
    if t < 60:
        return TimeUnit.SECONDS
    elif t < 60 * 60:
        return TimeUnit.MINUTES
    else:
        return TimeUnit.HOURS


def time_report(t, unit):
    if unit is None:
        unit = select_unit(t)
    if unit == TimeUnit.SECONDS:
        return f'{t} s'
    elif unit == TimeUnit.MINUTES:
        return f'{t / 60:.1f} m'
    elif unit == TimeUnit.HOURS:
        return f'{t / (60 * 60):.1f} h'
    else:
        raise ValueError(f'invalid unit "{unit}"')


def time_measure(level='info', unit=None):
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
    def __init__(self, name, level='info', unit=None, silent=False):
        self.name = name
        self.level = level
        self.unit = unit
        self.silent = silent
        self.sum = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time() - self.start
        self.sum += t
        if not self.silent:
            time_expr = time_report(t, self.unit)
            logger.log(levels[self.level], f'the code "{self.name}" executed in {time_expr}')

    def report_sum(self):
        time_expr = time_report(self.sum, self.unit)
        logger.log(levels[self.level], f'sum of execution time of "{self.name}" = {time_expr}')
