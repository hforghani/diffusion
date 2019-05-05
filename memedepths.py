# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import traceback
import time
from bson.objectid import ObjectId

import settings
from mongo import mongodb

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('memedepths')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Calculate meme depths.'

    def add_arguments(self, parser):
        parser.add_argument(
            "-c",
            "--continue",
            action="store_true",
            dest="do_continue",
            help="whether to continue from the last save point"
        )

    def handle(self, args):
        try:
            start = time.time()
            self.calc_depths(args.do_continue)
            logger.info('command done in %.2f min' % ((time.time() - start) / 60.0))
        except:
            logger.info(traceback.format_exc())
            raise

    def load_data(self, do_continue):
        i = 0
        depths = {}     # dictionary of meme id's to their depths
        tree_nodes = {}  # dictionary of meme id's to its tree nodes data. Each tree nodes data is a dictionary of user id's to its depth.

        if do_continue and os.path.exists('data/memedepths/i.json') and os.path.exists(
                'data/memedepths/depths.json') and os.path.exists('data/memedepths/tree_nodes.json'):
            logger.info('loading temp data ...')
            with open('data/memedepths/i.json') as f:
                i = json.load(f)['i']
            with open('data/memedepths/depths.json') as f:
                depths = json.load(f)
                depths = {ObjectId(key): value for key, value in depths.items()}
            with open('data/memedepths/tree_nodes.json') as f:
                tree_nodes = json.load(f)
                tree_nodes = {ObjectId(meme_id): {ObjectId(user_id): depth for user_id, depth in meme_data.items()} for
                              meme_id, meme_data in tree_nodes.items()}
            logger.info('data loaded')

        return i, depths, tree_nodes

    def save_data(self, i, depths, tree_nodes):
        if not os.path.exists('data/memedepths'):
            os.mkdir('data/memedepths')
        with open('data/memedepths/i.json', 'w') as f:
            json.dump({'i': i}, f)
        with open('data/memedepths/depths.json', 'w') as f:
            json.dump({str(key): value for key, value in depths.items()}, f, indent=4)
        with open('data/memedepths/tree_nodes.json', 'w') as f:
            json.dump({str(key): {str(user_id): value for user_id, value in nodes.items()} for key, nodes in
                       tree_nodes.items()}, f, indent=4)

    def calc_depths(self, do_continue=False):
        reshares = mongodb.reshares.find({},
            {'_id': 0, 'post_id': 1, 'reshared_post_id': 1, 'user_id': 1, 'ref_user_id': 1},
            no_cursor_timeout=True).sort('datetime')

        count = reshares.count()
        logger.info('number of all reshares: {}'.format(count))

        # If do_continue argument is true, continue from the last save point.

        i, depths, tree_nodes = self.load_data(do_continue)
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
                ref_memes = {m['meme_id'] for m in
                             mongodb.postmemes.find({'post_id': resh['reshared_post_id']}, {'meme_id': 1})}
                memes = {m['meme_id'] for m in
                         mongodb.postmemes.find({'post_id': resh['post_id']}, {'meme_id': 1})}
                common_memes = ref_memes & memes

                for meme_id in common_memes:
                    if meme_id not in depths:
                        depths[meme_id] = 1
                        tree_nodes[meme_id] = {src_uid: 0, dest_uid: 1}
                        # logger.info('meme {} has depth 1'.format(meme_id))

                    else:
                        nodes = tree_nodes[meme_id]
                        if src_uid not in nodes:
                            nodes[src_uid] = 0

                        if dest_uid not in nodes:
                            depth = nodes[src_uid] + 1
                            nodes[dest_uid] = depth
                            if depth > depths[meme_id]:
                                depths[meme_id] = depth
                                #logger.info('meme {} has now depth {}'.format(meme_id, depths[meme_id]))

            if i % step == 0:
                avg = (time.time() - t0) / (i - i0)
                rem = avg * (count - i)
                if rem > 60 * 60 * 48:
                    rem_str = '{:.0f} days'.format(rem / (60 * 60 * 24))
                elif rem > 60 * 60:
                    rem_str = '{:.0f} hours'.format(rem / (60 * 60))
                else:
                    rem_str = '{:.0f} minutes'.format(rem / 60)
                logger.info(
                    '{} reshares done. mean time: {:.0f} s per {}. estimated remaining time: {}'
                    .format(i, avg * step, step, rem_str))

            if i % save_step == 0:
                logger.info('saving temp data ...')
                self.save_data(i, depths, tree_nodes)

        logger.info('saving non-zero depths ...')
        for meme_id, depth in depths.items():
            mongodb.memes.find_one_and_update({'_id': meme_id}, {'$set': {'depth': depth}})

        logger.info('saving zero depths ...')
        mongodb.memes.update_many({'depth': None}, {'$set': {'depth': 0}})


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
