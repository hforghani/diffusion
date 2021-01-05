import csv
import os
import time
from pymongo import UpdateOne
from cascade.weibo import read_uidlist
from memm.db import DBManager
from settings import logger
import settings


def create_relations(relations_file, uidlist_file):
    t0 = time.time()

    uid_list = read_uidlist(uidlist_file)
    logger.info('reading relationships...')
    i = 0

    with open(relations_file, encoding='utf-8', errors='ignore') as f:
        f.readline()
        line = f.readline()
        relations = {}

        while line:
            line = line.strip().split()
            u1_i = int(line[0])
            u1 = uid_list[u1_i]
            n = int(line[1])
            parents = []
            for j in range(n):
                u2_i = int(line[2 + j * 2])
                u2 = uid_list[u2_i]
                parents.append(u2)
                relations.setdefault(u2, {'parents': [], 'children': []})['children'].append(u1)
            relations.setdefault(u1, {'parents': [], 'children': []})['parents'] = parents
            i += 1
            if i % 10000 == 0:
                logger.info('%d lines read', i)
            line = f.readline()
    logger.info('%d lines read', i)

    db = DBManager().db

    i = 0
    rel_to_insert = []
    for user_id in relations:
        rel = relations[user_id]
        rel_to_insert.append({'user_id': user_id, 'parents': rel['parents'], 'children': rel['children']})
        i += 1
        if i % 10000 == 0:
            logger.info('saving ...')
            db.relations.insert_many(rel_to_insert)
            logger.info('%d / %d relations saved', i, len(relations))
            rel_to_insert = []

    if rel_to_insert:
        logger.info('saving ...')
        db.relations.insert_many(rel_to_insert)
        logger.info('%d / %d relations saved', i, len(relations))
        rel_to_insert = []        

    logger.info('command done in %.2f min', (time.time() - t0) / 60.0)

if __name__ == '__main__':
    create_relations(settings.WEIBO_FOLLOWERS_PATH, settings.WEIBO_UIDLIST_PATH)
