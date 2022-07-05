import re


def is_number(text):
    if re.match('^\d+$', text):
        return True
    persian_digits = u'۰۱۲۳۴۵۶۷۸۹.٤٥٦1234567890'
    if sum([1 for ch in text if ch in persian_digits]) == len(text):
        return True
    return False


def columnize(alist, columns=3):
    """
    Generate a string containing the given strings organized by columns.
    """
    return '\n'.join(
        ''.join(f'{item:<30}' for item in alist[i:i + columns])
        for i in range(0, len(alist), columns)
    )
