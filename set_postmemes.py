import logging
from mongo import mongodb
import settings

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('set_postmemes')
logger.setLevel(settings.LOG_LEVEL)

if __name__ == '__main__':

    logger.info('extracting post datetimes ...')
    posts = mongodb.posts.find({}, ['_id', 'datetime'])

    logger.info('converting to dictionary ...')
    posts_map = {}
    i = 0
    count = mongodb.posts.count()
    for p in posts:
        posts_map[str(p['_id'])] = p['datetime']
        i += 1
        if i % 1000000 == 0:
            logger.info('%d%% done', i * 100 / count)
    logger.info('100%% done')

    logger.info('saving new postmemes with datetime ...')
    postmemes = []
    i = 37000000
    pm_count = mongodb.postmemes.count()
    for pm in mongodb.postmemes.find(no_cursor_timeout=True).skip(37000000):
        postmemes.append({'_id': pm['_id'], 'post_id': pm['post_id'], 'meme_id': pm['meme_id'],
                          'datetime': posts_map[str(pm['post_id'])]})
        i += 1
        if i % 1000000 == 0:
            mongodb.postmemes2.insert_many(postmemes)
            postmemes = []
            logger.info('%d%% done', i * 100 / pm_count)

    mongodb.postmemes2.insert_many(postmemes)
    logger.info('100%% done')