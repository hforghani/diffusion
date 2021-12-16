import sys

sys.path.append('.')

from db.managers import DBManager
from settings import logger

if __name__ == '__main__':
    db = DBManager().db
    post_ids = db.postmemes.find({'datetime': None}, {'post_id': 1, '_id': 0})
    post_ids_done = set()
    count = post_ids.count()
    i = 0
    last_percent = 0

    for p in post_ids:
        i += 1
        if p['post_id'] in post_ids_done:
            continue
        dt = db.posts.find_one({'_id': p['post_id']}, ['datetime'])['datetime']
        db.postmemes.update_many({'post_id': p['post_id']}, {'$set': {'datetime': dt}})
        post_ids_done.add(p['post_id'])
        if i * 100 // count > last_percent:
            last_percent = i * 100 // count
            logger.info('{}% done. datetime of {} postcascades set'.format(last_percent, i))

    logger.info('100% done')
