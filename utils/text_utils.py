# -*- coding: utf-8 -*-
import re


def unify_text(text):
    dic = {
        '۰': '0',
        '۱': '1',
        '۲': '2',
        '۳': '3',
        '۴': '4',
        '۵': '5',
        '۶': '6',
        '۷': '7',
        '۸': '8',
        '۹': '9',
        '.': '.',

        '١': '1',
        '٢': '2',
        '٣': '3',
        '٤': '4',
        '٥': '5',
        '٦': '6',
        '٧': '7',
        '٨': '8',
        '٩': '9',
        '٠': '0',

        'ك': 'ک',
        'دِ': 'د',
        'بِ': 'ب',
        'زِ': 'ز',
        'ذِ': 'ذ',
        'شِ': 'ش',
        'سِ': 'س',
        'ي': 'ی',
        'ى': 'ی'
    }
    new_text = text
    if isinstance(text, unicode):
        new_text = text.encode('utf8')

    pattern = '|'.join(map(re.escape, dic.keys()))
    new_text = re.sub(pattern, lambda m: dic[m.group()], new_text)

    if isinstance(text, unicode):
        new_text = new_text.decode('utf8')
    return new_text


def is_number(text):
    if re.match('^\d+$', text):
        return True
    persian_digits = u'۰۱۲۳۴۵۶۷۸۹.٤٥٦1234567890'
    if sum([1 for ch in text if ch in persian_digits]) == len(text):
        return True
    return False
