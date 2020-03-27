# -*- coding: utf-8 -*-
import argparse
import logging
import traceback
import time

from settings import mongodb
import settings
from cascade.weibo import create_users, create_roots, create_retweets, extract_follow_rels, calc_memes_values

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('readweibo')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Create database instances in MongoDB using weibo dataset.'

    def add_arguments(self, parser):
        parser.add_argument(
            "-u", "--users", type=str, dest="users_files", nargs='+',
            help="paths of user_profile(1 or 2).txt files in Weibo dataset"
        )
        parser.add_argument(
            "-r", "--roots", type=str, dest="roots_file",
            help="path of Root_Content.txt file in Weibo dataset"
        )
        parser.add_argument(
            "-t", "--retweets", type=str, dest="retweets_file",
            help="path of Retweet_Content.txt file in Weibo dataset"
        )
        parser.add_argument(
            "-f", "--followers", type=str, dest="followers_file",
            help="path of weibo_network.txt file (followers) in Weibo dataset"
        )
        parser.add_argument(
            "-i", "--uidlist", type=str, dest="uidlist_file",
            help="path of uidlist.txt file in Weibo dataset"
        )
        parser.add_argument(
            "-s", "--start", type=int, dest="start_index",
            help="determine which index of retweet data in the file Retweet_Content.txt to start from"
        )
        parser.add_argument(
            "-a", "--attributes", action="store_true", dest="set_attributes",
            help="just set attributes and ignore creating data"
        )
        parser.add_argument(
            "-c", "--clear", action="store_true", dest="clear",
            help="clear existing data and continue"
        )

    def handle(self, args):
        try:
            start = time.time()

            # Delete all data.
            if args.clear and not args.set_attributes:
                logger.info('======== deleting data ...')
                mongodb.postmemes.delete_many({})
                mongodb.reshares.delete_many({})
                mongodb.posts.delete_many({})
                mongodb.memes.delete_many({})
                mongodb.users.delete_many({})

            if not args.set_attributes:
                # Create users.
                if args.users_files:
                    logger.info('======== creating users ...')
                    users_map, user_ids = create_users(args.users_files)
                elif args.retweets_file:
                    logger.info('collecting users map ...')
                    users = mongodb.users.find({}, ['_id', 'username'])
                    users_map = {u['username']: u['_id'] for u in users if
                                 u['username'] is not None and u['username'] != ''}
                    users.rewind()
                    user_ids = {u['_id'] for u in users}

                # Create memes and their root posts.
                if args.roots_file:
                    logger.info('======== creating memes and roots ...')
                    memes_map = create_roots(args.roots_file)
                elif args.retweets_file:
                    logger.info('collecting posts map ...')
                    postmemes = mongodb.postmemes.find({}, ['post_id', 'meme_id'])
                    memes_map = {str(pm['post_id']): pm['meme_id'] for pm in postmemes}

                # Create retweet data and complete original posts fields.
                if args.retweets_file:
                    logger.info('======== creating retweets ...')
                    create_retweets(args.retweets_file, args.start_index, users_map, user_ids, memes_map)

                if args.followers_file and args.uidlist_file:
                    logger.info('======== creating following graph ...')
                    extract_follow_rels(args.followers_file, args.uidlist_file)

            # Set the meme count, first time, and last time attributes of memes.
            if args.set_attributes:
                logger.info('======== setting counts and publication times for the memes ...')
                calc_memes_values()

            logger.info('======== command done in %f min' % ((time.time() - start) / 60))
        except:
            logger.info(traceback.format_exc())
            raise


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
