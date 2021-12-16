import logging

import pymongo
from pymongo import IndexModel

import settings
from db.managers import DBManager


def handle(logger):
    db = DBManager().db
    logger.info('creating an index for cascades ...')
    db.memes.create_indexes([IndexModel('depth'), IndexModel('size')])
    logger.info('creating indexes for postcascades ...')
    db.postmemes.create_indexes([IndexModel('cascade_id'), IndexModel('post_id'), IndexModel('datetime')])
    logger.info('creating indexes for posts ...')
    db.posts.create_indexes([IndexModel('author_id'), IndexModel('datetime')])
    logger.info('creating indexes for reshares ...')
    db.reshares.create_indexes([IndexModel('post_id'), IndexModel('reshared_post_id'), IndexModel('datetime'),
                                IndexModel(
                                    [('user_id', pymongo.ASCENDING), ('ref_user_id', pymongo.ASCENDING)])])
    logger.info('creating indexes for relations ...')
    db.relations.create_indexes([IndexModel('user_id')])


if __name__ == '__main__':
    logging.basicConfig(format=settings.LOG_FORMAT)
    logger = logging.getLogger('create_indexes')
    logger.setLevel(settings.LOG_LEVEL)

    handle(logger)
