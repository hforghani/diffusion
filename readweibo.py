# -*- coding: utf-8 -*-
import argparse
import logging
import os
import re
import traceback
import time

from bson import SON
from bson.objectid import ObjectId
import pygtrie
from pymongo.errors import BulkWriteError
from pymongo.operations import UpdateOne

from mongo import mongodb
import settings
from utils.time_utils import str_to_datetime

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
                    users_map = self.create_users(args.users_files)
                else:
                    users = mongodb.users.find({}, ['_id', 'username'])
                    users_map = {u['username']: u['_id'] for u in users}

                # Create memes and their root posts.
                if args.roots_file:
                    logger.info('======== creating memes and roots ...')
                    memes_map = self.create_roots(args.roots_file)
                else:
                    postmemes = mongodb.postmemes.find({}, ['post_id', 'meme_id'])
                    memes_map = {str(pm['post_id']): pm['meme_id'] for pm in postmemes}

                # Create retweet data and complete original posts fields.
                if args.retweets_file:
                    logger.info('======== creating retweets ...')
                    self.create_retweets(args.retweets_file, args.start_index, users_map, memes_map)

            # Set the meme count, first time, and last time attributes of memes.
            if args.set_attributes:
                logger.info('======== setting counts and publication times for the memes ...')
                self.calc_memes_values()

            logger.info('======== command done in %f min' % ((time.time() - start) / 60))
        except:
            logger.info(traceback.format_exc())
            raise


    def create_roots(self, path):
        """
        Create memes and their root posts reading the file Root_Content.txt .
        :param path: path of the file Root_Content.txt
        """
        posts = []
        post_memes = []
        memes_map = {}
        i = 0

        with open(path, encoding='utf-8', errors='ignore') as f:
            line = f.readline()
            post_id = None

            while line:
                line = line.strip()

                if post_id is None:
                    if line and line[0] != '@' and line[:4] != 'link':
                        post_id = line
                else:
                    content = [int(index) for index in line.split(' ') if index]
                    post_id_obj = ObjectId('{:024d}'.format(int(post_id)))
                    posts.append({'_id': post_id_obj})
                    meme_id = mongodb.memes.insert_one({'text': content}).inserted_id
                    post_memes.append({'post_id': post_id_obj, 'meme_id': meme_id})
                    memes_map[post_id] = meme_id
                    post_id = None

                    i += 1
                    if i % 10000 == 0:
                        logger.info('%d posts read' % i)
                    if i % 100000 == 0:
                        mongodb.posts.insert_many(posts)
                        mongodb.postmemes.insert_many(post_memes)
                        logger.info('%d memes and their original posts created' % i)
                        posts = []
                        post_memes = []

                line = f.readline()

        if posts:
            mongodb.posts.insert_many(posts)
            mongodb.postmemes.insert_many(post_memes)
            logger.info('%d memes and their original posts created' % i)

        return memes_map


    def create_users(self, paths):
        """
        Create users reading the files user_profile1.txt and user_profile2.txt .
        :param paths: list of path of the files
        :returns map of usernames to user ids
        """
        users = []
        user_ids = {str(u['_id']) for u in mongodb.users.find({}, ['_id'])}
        users_map = {u['username']: u['_id'] for u in mongodb.users.find({}, ['_id', 'username'])}
        i = 0

        for path in paths:

            with open(path, encoding='GB18030', errors='replace') as f:
                # Skip first 15 lines due to the comments.
                for _ in range(15):
                    line = f.readline()

                while True:
                    try:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()
                        if not line:
                            continue

                        user_id = ObjectId('{:024d}'.format(int(line)))
                        for _ in range(7):
                            line = f.readline()
                        username = f.readline().strip()

                        if str(user_id) == '000000000000001477169412':
                            pass

                        if str(user_id) not in user_ids:
                            users.append({'_id': user_id, 'username': username})
                            users_map[username] = user_id
                            user_ids.add(str(user_id))

                            i += 1
                            if i % 10000 == 0:
                                logger.info('%d users read' % i)
                            if i % 100000 == 0:
                                mongodb.users.insert_many(users)
                                logger.info('%d users created' % i)
                                users = []

                        for _ in range(6):
                            line = f.readline()

                    except BulkWriteError as e:
                        print(e.details)
                        raise

            if users:
                mongodb.users.insert_many(users)
                logger.info('%d users created' % i)
                users = []

        return users_map


    # @profile
    def create_retweets(self, path, start_index, users_map, memes_map):
        i = 0
        t0 = time.time()
        memes_count = mongodb.memes.count()

        # if 'start_index' is specified, ignore lower indexes.
        ignoring = False
        if start_index:
            ignoring = True

        with open(path, encoding='GB18030', errors='replace') as f:

            while True:
                i += 1
                reshares = self.read_one_meme_reshares(f, users_map, memes_map)

                if (not ignoring or i == start_index) and i % 10000 == 0:
                    logger.info(
                        'saving %d post memes and %d reshares ...' % (len(post_memes), len(reshares)))
                    if post_memes or reshares:
                        mongodb.postmemes.insert_many(post_memes)
                        mongodb.reshares.insert_many(reshares)
                        post_memes = []
                        reshares = []
                        logger.info('time : %d s' % (time.time() - t0))
                        t0 = time.time()
                    logger.info('{:.0f}% done. processing from post number {} ...'.format(i / memes_count * 100, i))

                elif ignoring and i % 100000 == 0:
                    logger.info('ignoring posts: %d' % i)

                # Handle if it is in ignoring state.
                if ignoring:
                    if i <= start_index:
                        continue
                    else:
                        ignoring = False
                        t0 = time.time()

        # Save the remaining relations.
        logger.info(
            'saving %d post memes and %d reshares ...' % (len(post_memes), len(reshares)))
        mongodb.postmemes.insert_many(post_memes)
        mongodb.reshares.insert_many(reshares)

    def read_one_meme_reshares(self, f, users_map, memes_map):
        # Read root post data.
        line = f.readline()
        if line:
            line = line.strip()
        else:
            return None

        original_pid, original_uid, original_time, _ = line.split()
        original_pid = ObjectId('{:024d}'.format(int(original_pid)))
        original_uid = ObjectId('{:024d}'.format(int(original_uid)))
        original_time = str_to_datetime(original_time, '%Y-%m-%d-%H:%M:%S')

        mongodb.posts.update_one({'_id': original_pid},
                                 {'$set': {'datetime': original_time, 'author_id': original_uid}})
        meme_id = memes_map[str(original_pid)]
        uname_data = mongodb.users.find_one({'_id': original_uid}, {'_id': False, 'username': True})
        original_uname = uname_data['username'] if uname_data else str(original_uid)
        print('\nreading meme {}'.format(str(meme_id)))

        # Read retweets number.
        line = f.readline().strip()
        retweet_num = int(line)

        reshares = []
        ret_lists = []

        for i in range(retweet_num):
            ret_list, ret_time = self.read_one_reshare(f, meme_id, users_map, original_uname)
            ret_lists.append((ret_list, ret_time))
            #reshares.append(reshare)

        ret_lists = sorted(ret_lists, key=lambda x: x[1])
        for item in ret_lists:
            print(str(item[1]) + ' : ' + ' -> '.join(item[0]))

        return reshares


    def read_one_reshare(self, f, meme_id, users_map, original_uname):
        # Read retweet data.
        line = f.readline()
        if not line:
            return None
        line = line.strip()

        retweet_uid, retweet_time, retweet_pid = line.split()
        retweet_pid = ObjectId('{:024d}'.format(int(retweet_pid)))
        retweet_uid = ObjectId('{:024d}'.format(int(retweet_uid)))
        retweet_time = str_to_datetime(retweet_time, '%Y-%m-%d-%H:%M:%S')

        # Skip retweet content.
        f.readline()

        # Skip mention line if exist.
        last_pos = f.tell()
        line = f.readline().strip()
        if line[0] == '@':
            last_pos = f.tell()
            line = f.readline().strip()

        uname_data = mongodb.users.find_one({'_id': retweet_uid}, {'_id': False, 'username': True})
        uname = uname_data['username'] if uname_data else str(retweet_uid)

        # Read retweet list if exists.
        if line[:7] == 'retweet':
            ret_list = [original_uname] + line[8:].split() + [uname]
            #anc_ids = [users_map[uname] for uname in ancestors]
            last_pos = f.tell()
            line = f.readline().strip()
        else:
            ret_list = [original_uname, uname]

        if line[:4] != 'link':
            f.seek(last_pos)

        #reshare = {'post_id': retweet_pid, 'reshared_post_id': parent_pid, 'datetime': retweet_time,
        #           'user_id': retweet_uid, 'ref_user_id': parent_uid, 'ref_datetime': original_time}
        reshare = {}

        return ret_list, retweet_time


    def calc_memes_values(self):
        count = mongodb.memes.count()
        save_step = 10 ** 6

        logger.info('query of meme counts ...')
        meme_counts = mongodb.postmemes.aggregate([{'$group': {'_id': '$meme_id', 'count': {'$sum': 1}}}],
                                                  allowDiskUse=True)

        logger.info('saving ...')
        operations = []
        i = 0
        for doc in meme_counts:
            operations.append(UpdateOne({'_id': doc['_id']}, {'$set': {'count': doc['count']}}))
            i += 1
            if i % save_step == 0:
                mongodb.memes.bulk_write(operations)
                operations = []
                logger.info('%d%% done', i * 100 / count)
        mongodb.memes.bulk_write(operations)

        logger.info('query of first times ...')
        first_times = mongodb.postmemes.aggregate([{'$group': {'_id': '$meme_id', 'first': {'$min': '$datetime'}}}],
                                                  allowDiskUse=True)

        logger.info('saving ...')
        operations = []
        i = 0
        for doc in first_times:
            operations.append(UpdateOne({'_id': doc['_id']}, {'$set': {'first_time': doc['first']}}))
            i += 1
            if i % save_step == 0:
                mongodb.memes.bulk_write(operations)
                operations = []
                logger.info('%d%% done', i * 100 / count)
        mongodb.memes.bulk_write(operations)

        logger.info('query of last times ...')
        last_times = mongodb.postmemes.aggregate([{'$group': {'_id': '$meme_id', 'last': {'$max': '$datetime'}}}],
                                                 allowDiskUse=True)

        logger.info('saving ...')
        operations = []
        i = 0
        for doc in last_times:
            operations.append(UpdateOne({'_id': doc['_id']}, {'$set': {'last_time': doc['last']}}))
            i += 1
            if i % save_step == 0:
                mongodb.memes.bulk_write(operations)
                operations = []
                logger.info('%d%% done', i * 100 / count)
        mongodb.memes.bulk_write(operations)


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
