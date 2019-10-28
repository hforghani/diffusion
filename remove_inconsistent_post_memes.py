import logging
from settings import mongodb
import settings

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('remove_inconsistent_post_memes')
logger.setLevel(settings.LOG_LEVEL)

i = 0
logger.info('counting posts ...')
count = mongodb.posts.find({'datetime': None}).count()
logger.info('removing inconsistent posts ...')

for p in mongodb.posts.find({'datetime': None}, ['_id'], no_cursor_timeout=True):
    mongodb.postmemes.remove({'post_id': p['_id']})
    i += 1
    if i % 10000 == 0:
        logger.info('%d posts done: %.1f%%', i, i / count * 100)
