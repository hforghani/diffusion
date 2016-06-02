# -*- coding: utf-8 -*-

import _strptime  # Don't remove this line. It avoids import errors!
from datetime import timedelta, datetime
from django.utils.timezone import get_default_timezone_name
import pytz

DT_FORMAT = '%Y-%m-%d %H:%M:%S'


def get_scale_for_chart(min_datetime, max_datetime):
    """
    Get time scale for horizontal axis of charts.
    """
    dist = max_datetime - min_datetime
    if dist <= timedelta(days=3):
        return 'hour'
    elif dist <= timedelta(days=60):
        return 'day'
    elif dist <= timedelta(weeks=30):
        return 'week'
    elif dist <= timedelta(days=2000):
        return 'month'
    else:
        return 'year'


def get_scale_for_evol(min_datetime, max_datetime):
    """
    Get time scale for evolution processes (such as community evolution). The difference is that time ticks are larger.
    """
    dist = max_datetime - min_datetime
    if dist <= timedelta(hours=48):
        return 'hour'
    elif dist <= timedelta(days=21):
        return 'day'
    elif dist <= timedelta(weeks=20):  # less than 5 months
        return 'week'
    elif dist <= timedelta(days=1100):  # less than 2 years
        return 'month'
    else:
        return 'year'


def time_delta_by_name(scale):
    if scale == 'year':
        return timedelta(days=365)
    if scale == 'month':
        return timedelta(days=30)
    if scale == 'week':
        return timedelta(weeks=1)
    if scale == 'day':
        return timedelta(days=1)
    if scale == 'hour':
        return timedelta(hours=1)
    return None


def timeline_data(datetimes, min_dt=None, max_dt=None):
    res = None

    if datetimes:
        datetimes.sort()
        if not min_dt:
            min_dt = datetimes[0]
        if not max_dt:
            max_dt = datetimes[-1]
        scale = get_scale_for_chart(min_dt, max_dt)
        res = {'scale': scale, 'counts': []}
        delta = time_delta_by_name(scale)
        dt = min_dt
        i = 0
        while dt <= max_dt + delta:
            count = 0
            while i < len(datetimes) and datetimes[i] < dt:
                count += 1
                i += 1
            res['counts'].append({'datetime': dt.strftime(DT_FORMAT), 'count': count})
            dt += delta
        if res['counts']:
            del res['counts'][0]
    elif max_dt and min_dt:
        scale = get_scale_for_chart(min_dt, max_dt)
        delta = time_delta_by_name(scale)
        res = {'scale': scale, 'counts': [0] * int((max_dt - min_dt) / delta)}

    return res


def localize(dt):
    return pytz.timezone(get_default_timezone_name()).localize(dt)


def str_to_datetime(datetime_str, dt_format=None):
    if not dt_format:
        dt_format = DT_FORMAT
    dt = datetime.strptime(datetime_str, dt_format)
    return localize(dt)


def shamsi_month_num(month_name):
    months = [u'فروردین', u'اردیبهشت', u'خرداد', u'تیر', u'مرداد', u'شهریور', u'مهر', u'آبان', u'آذر', u'دی', u'بهمن',
              u'اسفند']
    index = months.index(month_name)
    return index + 1 if index != -1 else None