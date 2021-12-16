# -*- coding: utf-8 -*-
import argparse
import os
import traceback
from random import shuffle
import time
import numpy as np
from bson import ObjectId

from cascade.models import Project
from db.managers import DBManager
from settings import logger, BASE_PATH


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
            help="number of users which we want to sample their cascades",
        )
        parser.add_argument(
            "-d",
            "--mindepth",
            type=int,
            dest="min_depth",
            default=0,
            help="minimum depth of cascade trees of cascades",
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

            db = DBManager().db

            if args.sample_num:
                # if args.users_num:
                #     # Sample a limited number of user id's and get their cascades. Then sample cascades from this set.
                #     logger.info('sampling users ...')
                #     user_samples = [u['_id'] for u in mongodb.users.aggregate([{'$sample': {'size': args.users_num}},
                #                                                                {'$project': {'_id': 1}}])]
                #     logger.info('sampling cascades ...')
                #     post_ids = [p['_id'] for p in mongodb.posts.find({'author_id': {'$in': user_samples}}, ['_id'])]
                #     user_cascades = [pm['cascade_id'] for pm in
                #                   mongodb.postcascades.find({'post_id': {'$in': post_ids}}, {'_id': 0, 'cascade_id': 1})]
                #     query = {'_id': {'$in': user_cascades}}
                #     if args.min_depth:
                #         query['depth'] = {'$gte': args.min_depth}
                #     cascade_ids = [m['_id'] for m in mongodb.cascades.aggregate([
                #         {'$match': query},
                #         {'$sample': {'size': args.sample_num}},
                #         {'$project': {'_id': 1}}
                #     ])]
                # else:
                #     # Sample sample_num cascades with minimum depth if given.
                #     logger.info('sampling cascades ...')
                #     query = {}
                #     if args.min_depth:
                #         query = {'depth': {'$gte': args.min_depth}}
                #     cascade_ids = [m['_id'] for m in mongodb.cascades.aggregate([
                #         {'$match': query},
                #         {'$sample': {'size': args.sample_num}},
                #         {'$project': {'_id': 1}}
                #     ])]

                selected = np.load(os.path.join(BASE_PATH, 'data/weibo_cascade_labels2.npy'))
                logger.info('fetching all cascade ids ...')
                cursor = db.memes.find({}, ['_id'], no_cursor_timeout=True).sort('_id')
                all_cascade_ids = [m['_id'] for m in cursor]
                cursor.close()
                selected_cascades = np.array([str(mid) for mid in all_cascade_ids])[selected]
                selected_cascades = [ObjectId(mid) for mid in selected_cascades]
                query = {'_id': {'$in': selected_cascades}}
                if args.min_depth:
                    query['depth'] = {'$gte': args.min_depth}
                logger.info('sampling cascades ...')
                cascade_ids = [m['_id'] for m in db.memes.aggregate([
                    {'$match': query},
                    {'$sample': {'size': args.sample_num}},
                    {'$project': {'_id': 1}}
                ])]

            else:
                # Get all cascades.
                # TODO: Load selected cascades.
                logger.info('sampling cascades ...')
                query = {}
                if args.min_depth:
                    query = {'depth': {'$gte': args.min_depth}}
                cascade_ids = [m['_id'] for m in db.memes.find(query, ['_id'])]
                shuffle(cascade_ids)

            if not cascade_ids:
                raise Exception('no cascades sampled; change the command arguments')
            elif args.sample_num and len(cascade_ids) < args.sample_num:
                logger.warn('number of sampled cascades is less than "number" argument')

            # Separate training and test sets.
            ratio = args.ratio
            train_num = int(ratio * len(cascade_ids))
            cascade_ids = [str(m_id) for m_id in cascade_ids]
            train_set = cascade_ids[:train_num]
            test_set = cascade_ids[train_num:]

            project = Project(project_name)
            project.save_sets(train_set, [], test_set)

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
