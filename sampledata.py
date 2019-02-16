# -*- coding: utf-8 -*-
import argparse
import logging
import traceback
from random import shuffle
import time
import numpy as np

import settings
from cascade.models import Project
from mongo import mongodb

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('sampledata')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Sample a subset of data and separate training and test sets and save them into the file'

    def add_arguments(self, parser):
        parser.add_argument(
            "-n",
            "--number",
            type=int,
            dest="sample_num",
            help="number of data samples consisting training and test sets",
        )
        parser.add_argument(
            "-u",
            "--usersnum",
            type=int,
            dest="users_num",
            help="number of users which we want to sample their memes",
        )
        parser.add_argument(
            "-r",
            "--ratio",
            type=float,
            dest="ratio",
            default=2.0 / 3,
            help="ratio of number of training set to number of all samples",
        )
        parser.add_argument(
            "-p",
            "--project",
            type=str,
            dest="project",
            help="project name",
        )

    def handle(self, args):
        try:
            start = time.time()

            # Get project of raise exception.
            project_name = args.project
            if project_name is None:
                raise Exception('project not specified')

            # Load meme and user id's.
            min_depth = 1

            if args.sample_num:
                meme_ids = []
                # Repeat sampling if the sampled users do not have enough memes.
                while not meme_ids:
                    # Sample user id's and get their memes. Sample memes from this set.
                    logger.info('sampling data ...')
                    sample_num = args.sample_num
                    users_num = args.users_num if args.users_num else sample_num * 10
                    user_samples = [u['_id'] for u in mongodb.aggregate([{'$sample': users_num},
                                                                         {'$project': ['_id']}])]
                    # user_ids = [u['_id'] for u in mongodb.find({}, ['_id'])]
                    # user_samples = list(np.random.choice(user_ids, users_num, replace=False))
                    post_ids = [p['_id'] for p in mongodb.posts.find({'author_id': {'$in': user_samples}}, ['_id'])]
                    user_memes = [pm['meme_id'] for pm in
                                  mongodb.memes.find({'post_id': {'$in': post_ids}}, {'_id': 0, 'meme_id': 1})]
                    meme_ids = mongodb.memes.aggregate([
                        {'$match': {'_id': {'$in': user_memes}}},
                        {'$sample': sample_num},
                        {'$project': ['_id']}
                    ])
                    # meme_ids = list(np.random.choice(user_memes, sample_num, replace=False))
            else:
                # Get all memes.
                logger.info('sampling data ...')
                meme_ids = [m['_id'] for m in mongodb.memes.find({'depth': {'$ge': min_depth}}, ['_id'])]
                shuffle(meme_ids)

            # Separate training and test sets.
            ratio = args.ratio
            train_num = int(ratio * len(meme_ids))
            meme_ids = [int(m_id) for m_id in meme_ids]
            train_set = meme_ids[:train_num]
            test_set = meme_ids[train_num:]

            project = Project(project_name)
            project.save_data(test_set, train_set)

            logger.info('command done in %f min' % ((time.time() - start) / 60))
        except:
            logger.info(traceback.format_exc())
            raise


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
