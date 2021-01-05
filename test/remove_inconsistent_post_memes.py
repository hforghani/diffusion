import logging

from memm.db import DBManager
import settings

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('remove_inconsistent_post_memes')
logger.setLevel(settings.LOG_LEVEL)

i = 0
logger.info('counting posts ...')
db = DBManager().db
count = db.posts.find({'datetime': None}).count()
logger.info('removing inconsistent posts ...')

cursor = db.posts.find({'datetime': None}, ['_id'], no_cursor_timeout=True)
for p in cursor:
    db.postmemes.remove({'post_id': p['_id']})
    i += 1
    if i % 10000 == 0:
        logger.info('%d posts done: %.1f%%', i, i / count * 100)
cursor.close()