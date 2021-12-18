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
logger = logging.getLogger('cascade_depths')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Calculate cascade depths.'

    def add_arguments(self, parser):
        parser.add_argument(
            "-c",
            "--continue",
            action="store_true",
            dest="do_continue",
            help="whether to continue from the last save point"
        )

    @time_measure()
    def handle(self, args):
        try:
            self.calc_depths(args.do_continue)
        except:
            logger.info(traceback.format_exc())
            raise

    def load_data(self, do_continue):
        i = 0
        depths = {}     # dictionary of cascade id's to their depths
        tree_nodes = {}  # dictionary of cascade id's to their tree nodes data. Each tree nodes data is a dictionary of user id's to their depths.

        if do_continue and os.path.exists('data/cascade_depths/i.json') and os.path.exists(
                'data/cascade_depths/depths.json') and os.path.exists('data/cascade_depths/tree_nodes.json'):
            logger.info('loading temp data ...')
            with open('data/cascade_depths/i.json') as f:
                i = json.load(f)['i']
            with open('data/cascade_depths/depths.json') as f:
                depths = json.load(f)
                depths = {ObjectId(key): value for key, value in depths.items()}
            with open('data/cascade_depths/tree_nodes.json') as f:
                tree_nodes = json.load(f)
                tree_nodes = {ObjectId(cascade_id): {ObjectId(user_id): depth for user_id, depth in cascade_data.items()} for
                              cascade_id, cascade_data in tree_nodes.items()}
            logger.info('data loaded')

        return i, depths, tree_nodes

    def save_data(self, i, depths, tree_nodes):
        if not os.path.exists('data/cascade_depths'):
            os.mkdir('data/cascade_depths')
        with open('data/cascade_depths/i.json', 'w') as f:
            json.dump({'i': i}, f)
        with open('data/cascade_depths/depths.json', 'w') as f:
            json.dump({str(key): value for key, value in depths.items()}, f, indent=4)
        with open('data/cascade_depths/tree_nodes.json', 'w') as f:
            json.dump({str(key): {str(user_id): value for user_id, value in nodes.items()} for key, nodes in
                       tree_nodes.items()}, f, indent=4)

    def calc_depths(self, do_continue=False):
        db = DBManager().db
        count = db.reshares.count()
        logger.info('number of all reshares: {}'.format(count))

        reshares = db.reshares.find({},
            {'_id': 0, 'post_id': 1, 'reshared_post_id': 1, 'user_id': 1, 'ref_user_id': 1},
            no_cursor_timeout=True).sort('datetime')

        # If do_continue argument is true, continue from the last save point.

        i, depths, tree_nodes = self.load_data(do_continue)
        """
        i : Number of iterated reshares 
        depths: dictionary of cascade id's to their depths
        tree_nodes: dictionary of cascade id's to their tree nodes data. Each tree nodes data is a dictionary of 
                    user id's to their depths.
        """
        if i > 0:
            reshares = reshares.skip(i)

        t0 = time.time()
        i0 = i
        step = 10 ** 5
        save_step = 2 * 10 ** 6

        for resh in reshares:
            i += 1
            src_uid, dest_uid = resh['ref_user_id'], resh['user_id']

            if src_uid != dest_uid:
                ref_cascades = {m['cascade_id'] for m in
                             db.postcascades.find({'post_id': resh['reshared_post_id']}, {'cascade_id': 1})}
                cascades = {m['cascade_id'] for m in
                         db.postcascades.find({'post_id': resh['post_id']}, {'cascade_id': 1})}
                common_cascades = ref_cascades & cascades

                for cascade_id in common_cascades:
                    if cascade_id not in depths:
                        depths[cascade_id] = 1
                        tree_nodes[cascade_id] = {src_uid: 0, dest_uid: 1}
                        # logger.info('cascade {} has depth 1'.format(cascade_id))

                    else:
                        nodes = tree_nodes[cascade_id]
                        if src_uid not in nodes:
                            nodes[src_uid] = 0

                        if dest_uid not in nodes:
                            depth = nodes[src_uid] + 1
                            nodes[dest_uid] = depth
                            if depth > depths[cascade_id]:
                                depths[cascade_id] = depth
                                #logger.info('cascade {} has now depth {}'.format(cascade_id, depths[cascade_id]))

            if i % step == 0:
                logger.info('%d reshares done', i)

            if i % save_step == 0:
                logger.info('saving temp data ...')
                self.save_data(i, depths, tree_nodes)
                logger.info('temp data saved')
                avg = (time.time() - t0) / (i - i0)
                rem = avg * (count - i)
                if rem > 60 * 60 * 48:
                    rem_str = '{:.0f} days'.format(rem / (60 * 60 * 24))
                elif rem > 60 * 60:
                    rem_str = '{:.0f} hours'.format(rem / (60 * 60))
                else:
                    rem_str = '{:.0f} minutes'.format(rem / 60)
                logger.info(f'mean time: {(avg * step):.0f} s per {step}. estimated remaining time: {rem_str}')
                t0 = time.time()
                i0 = i

        reshares.close()

        logger.info('saving non-zero depths ...')
        for cascade_id, depth in depths.items():
            db.cascades.find_one_and_update({'_id': cascade_id}, {'$set': {'depth': depth}})

        logger.info('saving zero depths ...')
        db.cascades.update_many({'depth': None}, {'$set': {'depth': 0}})


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
