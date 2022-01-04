import sys

sys.path.append('.')

from db.managers import DBManager
from settings import logger

if __name__ == '__main__':
    db = DBManager('weibo').db
    count = db.posts.count()
    posts = db.posts.find({}, ['_id', 'author_id'], no_cursor_timeout=True)
    logger.info('setting authors of %d posts in postcascades collection ...', count)
    i = 0

    for p in posts:
        i += 1
        db.postcascades.update_many({'post_id': p['_id']}, {'$set': {'author_id': p['author_id']}})
        if i % 1000 == 0:
            logger.info('%d%% done. author_id of %d posts set', i / count * 100, i)

    logger.info('100% done')
