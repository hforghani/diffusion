import logging

import pymongo
from pymongo import IndexModel

import settings
from settings import mongodb


def handle(logger):
    logger.info('creating an index for memes ...')
    mongodb.memes.create_index('depth')
    logger.info('creating indexes for postmemes ...')
    mongodb.postmemes.create_indexes([IndexModel('meme_id'), IndexModel('post_id'), IndexModel('datetime')])
    logger.info('creating indexes for posts ...')
    mongodb.posts.create_indexes([IndexModel('author_id'), IndexModel('datetime')])
    logger.info('creating indexes for reshares ...')
    mongodb.reshares.create_indexes([IndexModel('post_id'), IndexModel('reshared_post_id'), IndexModel('datetime'),
                                     IndexModel(
                                         [('user_id', pymongo.ASCENDING), ('ref_user_id', pymongo.ASCENDING)])])
    logger.info('creating indexes for relations ...')
    mongodb.relations.create_indexes([IndexModel('user_id')])


if __name__ == '__main__':
    logging.basicConfig(format=settings.LOG_FORMAT)
    logger = logging.getLogger('create_indexes')
    logger.setLevel(settings.LOG_LEVEL)

    handle(logger)