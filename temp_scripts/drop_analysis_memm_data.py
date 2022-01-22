from pymongo import MongoClient
import sys

sys.path.append('.')

from cascade.enum import Method
from settings import logger

dataset = 'weibo'
projects_num = 100
methods = [Method.LONG_PARENT_SENS_TD_MEMM,
           Method.PARENT_SENS_TD_MEMM,
           Method.REDUCED_TD_MEMM,
           Method.TD_MEMM,
           Method.REDUCED_BIN_MEMM,
           Method.BIN_MEMM,
           Method.ASLT,
           Method.AVG]

client = MongoClient()

for i in range(1, projects_num + 1):
    project_name = f'{dataset}-analysis-{i}'
    for method in methods:
        evid_db_name = f'{dataset}_{method.value}_evid_{project_name}'
        memm_db_name = f'{dataset}_{method.value}_{project_name}'
        client.drop_database(evid_db_name)
        client.drop_database(memm_db_name)
    logger.info('data of project %s dropped', project_name)
