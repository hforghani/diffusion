import os
import pprint

from matplotlib import pyplot
from pymongo import MongoClient
import numpy as np
import sys

sys.path.append('.')

from local_settings import BASE_PATH

db_name = 'weibo'
range_max = None
bins_num = 100

client = MongoClient()
db = client[db_name]
data = db.relations.aggregate([
    {"$project": {"parents_count": {"$size": "$parents"}}}
])
pcounts = [d['parents_count'] for d in data]
counts, bins = np.histogram(pcounts, bins=bins_num, range=(0, range_max) if range_max else None)
pyplot.bar(bins[:-1], counts, width=bins[-1] / bins_num)
pyplot.savefig(os.path.join(BASE_PATH, 'data', f'{db_name}_parents_hist.png'))
