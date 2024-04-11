import argparse

import pymongo
from pymongo import IndexModel

from db.managers import DBManager
from settings import logger


def handle(db_name):
    db = DBManager(db_name).db
    logger.info('creating an index for cascades ...')
    db.cascades.create_indexes([IndexModel('depth'), IndexModel('size')])
    logger.info('creating indexes for postcascades ...')
    db.postcascades.create_indexes([IndexModel('cascade_id'), IndexModel('post_id'), IndexModel('datetime')])
    logger.info('creating indexes for posts ...')
    db.posts.create_indexes([IndexModel('author_id'), IndexModel('datetime')])
    logger.info('creating indexes for reshares ...')
    db.reshares.create_indexes([IndexModel('post_id'), IndexModel('reshared_post_id'), IndexModel('datetime'),
                                IndexModel(
                                    [('user_id', pymongo.ASCENDING), ('ref_user_id', pymongo.ASCENDING)])])
    # logger.info('creating indexes for relations ...')
    # db.relations.create_indexes([IndexModel('user_id')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Show statistics of the cascades between a min and max user count')
    parser.add_argument('-d', '--db', required=True, help="db name")
    args = parser.parse_args()
    handle(args.db)
