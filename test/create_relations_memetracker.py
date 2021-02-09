from settings import logger
from utils.db import DBManager
from utils.time_utils import time_measure


@time_measure()
def create_relations():
    db = DBManager().db
    relations = {}
    i = 0

    for resh in db.reshares.find({}, no_cursor_timeout=True):
        if 'ref_user_id' in resh and 'user_id' in resh:
            parent = resh['ref_user_id']
            child = resh['user_id']
            if child is not None and parent is not None:
                relations.setdefault(parent, {'parents': set(), 'children': set()})
                relations[parent]['children'].add(child)
                relations.setdefault(child, {'parents': set(), 'children': set()})
                relations[child]['parents'].add(parent)
        i += 1
        if i % 1000 == 0:
            logger.info('%d reshares done', i)

    logger.info('creating %d relations ...', len(relations))
    relations = [{'user_id': uid,
                  'parents': list(data['parents']),
                  'children': list(data['children'])} for uid, data in relations.items()]
    db.relations.insert_many(relations)
    logger.info('done')


if __name__ == '__main__':
    create_relations()
