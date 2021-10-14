# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import traceback
import time
from bson.objectid import ObjectId

import settings
from db.managers import DBManager
from utils.time_utils import time_measure

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('cascade_depth_single')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Calculate a cascade depth.'

    def add_arguments(self, parser):
        parser.add_argument(
            'cascade_id',
            help="cascade id"
        )

    @time_measure()
    def handle(self, args):
        try:
            self.calc_depths(args.cascade_id)
        except:
            logger.info(traceback.format_exc())
            raise

    def calc_depths(self, cascade_id):
        db = DBManager().db
        count = db.reshares.count()
        logger.info('number of all reshares: {}'.format(count))

        post_ids = [pm['post_id'] for pm in
                    db.postmemes.find({'meme_id': ObjectId(cascade_id)}, {'_id': 0, 'post_id': 1})]

        reshares = db.reshares.find({'post_id': {'$in': post_ids}, 'reshared_post_id': {'$in': post_ids}},
                                    {'_id': 0, 'user_id': 1, 'ref_user_id': 1},
                                    no_cursor_timeout=True).sort('datetime')

        # If do_continue argument is true, continue from the last save point.

        depth = 0  # current cascade depth
        node_depths = {}  # dictionary of user id's to their depths.

        i = 0
        t0 = time.time()
        step = 10 ** 5

        for resh in reshares:
            i += 1
            src_uid, dest_uid = resh['ref_user_id'], resh['user_id']

            if src_uid != dest_uid:
                if src_uid not in node_depths:
                    node_depths[src_uid] = 0

                if dest_uid not in node_depths:
                    dest_depth = node_depths[src_uid] + 1
                    node_depths[dest_uid] = dest_depth
                    if dest_depth > depth:
                        depth = dest_depth
                        logger.info('meme %s has now depth %d', cascade_id, depth)

            if i % step == 0:
                avg = (time.time() - t0) / i
                rem = avg * (count - i)
                if rem > 60 * 60 * 48:
                    rem_str = '{:.0f} days'.format(rem / (60 * 60 * 24))
                elif rem > 60 * 60:
                    rem_str = '{:.0f} hours'.format(rem / (60 * 60))
                else:
                    rem_str = '{:.0f} minutes'.format(rem / 60)
                logger.info(
                    f'{i} reshares done. mean time: {(avg * step):.0f} s per {step}. estimated remaining time: {rem_str}')

        db.memes.find_one_and_update({'_id': cascade_id}, {'$set': {'depth': depth}})


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
