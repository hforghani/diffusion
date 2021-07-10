import sys

sys.path.append('.')

from db.managers import DBManager
from settings import logger

if __name__ == '__main__':
    db = DBManager().db
    logger.info('extracting postmemes without author ...')
    post_ids = db.postmemes.find({'author': None}, {'post_id': 1, '_id': 0})
    post_ids_done = set()
    count = post_ids.count()
    logger.info('done')
    logger.info('setting authors ...')
    i = 0

    for p in post_ids:
        i += 1
        if p['post_id'] in post_ids_done:
            continue
        author_id = db.posts.find_one({'_id': p['post_id']}, {'author_id': 1, '_id': 0})['c']
        db.postmemes.update_many({'post_id': p['post_id']}, {'$set': {'author_id': author_id}})
        post_ids_done.add(p['post_id'])
        if i % 1000 == 0:
            logger.info('{:.0f}% done. author_id of {} postmemes set'.format(i / count * 100, i))

    logger.info('100% done')
