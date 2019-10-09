# -*- coding: utf-8 -*-
import argparse
import logging
import traceback
import time

from bson.objectid import ObjectId
from datetime import timedelta
import pymongo
from pymongo.errors import BulkWriteError
from pymongo.operations import UpdateOne, IndexModel

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
            "-i", "--index", action="store_true", help="create the indexes"
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
                    users_map, user_ids = self.create_users(args.users_files)
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
                    memes_map = self.create_roots(args.roots_file)
                elif args.retweets_file:
                    logger.info('collecting posts map ...')
                    postmemes = mongodb.postmemes.find({}, ['post_id', 'meme_id'])
                    memes_map = {str(pm['post_id']): pm['meme_id'] for pm in postmemes}

                # Create retweet data and complete original posts fields.
                if args.retweets_file:
                    logger.info('======== creating retweets ...')
                    self.create_retweets(args.retweets_file, args.start_index, users_map, user_ids, memes_map)

            # Create the indexes.
            if args.index:
                logger.info('======== creating indexes ...')
                self.create_indexes()

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
        user_ids = {u['_id'] for u in mongodb.users.find({}, ['_id'])}
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

                        if user_id not in user_ids:
                            users.append({'_id': user_id, 'username': username})
                            if username:
                                users_map[username] = user_id
                            user_ids.add(user_id)

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

        return users_map, user_ids


    # @profile
    def create_retweets(self, path, start_index, users_map, user_ids, memes_map):
        i = 0
        t0 = time.time()
        memes_count = mongodb.memes.count()
        post_memes = []
        reshares = []

        # if 'start_index' is specified, ignore lower indexes.
        ignoring = False
        if start_index:
            ignoring = True

        with open(path, encoding='GB18030', errors='replace') as f:

            while True:
                i += 1
                resh_list, pm_list = self.read_one_meme_reshares(f, users_map, user_ids, memes_map, ignoring)
                if resh_list is None and pm_list is None:
                    break
                reshares.extend(resh_list)
                post_memes.extend(pm_list)

                if (not ignoring or i == start_index) and (len(post_memes) > 10000 or len(reshares) > 10000):
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

                elif ignoring and i % 100 == 0:
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

    def read_one_meme_reshares(self, f, users_map, user_ids, memes_map, ignoring=False):
        # Read root post data.
        line = None
        while line is None or line == '\n':
            line = f.readline()
            if not line:
                return None, None

        original_pid, original_uid, original_time, _ = line.split()
        original_pid = ObjectId('{:024d}'.format(int(original_pid)))
        original_uid = ObjectId('{:024d}'.format(int(original_uid)))
        original_time = str_to_datetime(original_time, '%Y-%m-%d-%H:%M:%S')

        if original_uid not in user_ids:
            mongodb.users.insert_one({'_id': original_uid, 'username': None})
            user_ids.append(original_uid)

        if not ignoring:
            mongodb.posts.update_one({'_id': original_pid},
                                     {'$set': {'datetime': original_time, 'author_id': original_uid}})
            mongodb.postmemes.update_one({'post_id': original_pid}, {'$set': {'datetime': original_time}})
        meme_id = memes_map[str(original_pid)]

        # Read retweets number.
        line = f.readline().strip()
        retweet_num = int(line)

        meme_reshares = []
        meme_postmemes = []
        #resh_pairs = set()
        posts_map = {str(original_uid): {'_id': original_pid, 'datetime': original_time}}

        for i in range(retweet_num):
            reshares, post_memes = self.read_one_reshare_seq(f, meme_id, original_uid, users_map, posts_map, ignoring)
            meme_reshares.extend(reshares)
            meme_postmemes.extend(post_memes)

        return meme_reshares, meme_postmemes


    def read_one_reshare_seq(self, f, meme_id, original_uid, users_map, posts_map, ignoring=False):
        # Read retweet data.
        line = None
        while line is None or line == '\n':
            line = f.readline()
            if not line:
                return None, None

        try:
            retweet_uid, retweet_time, retweet_pid = line.split()
        except ValueError:
            raise
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

        # Read retweet list if exists.
        if line[:7] == 'retweet':
            ret_list = line[8:].split()
            uid_list = [original_uid]

            for uname in ret_list:
                if uname in users_map:
                    uid_list.append(users_map[uname])
                else:
                    user_id = mongodb.users.insert_one({'username': uname}).inserted_id
                    uid_list.append(user_id)

            uid_list.append(retweet_uid)

            last_pos = f.tell()
            line = f.readline().strip()

        else:
            uid_list = [original_uid, retweet_uid]

        if line[:4] != 'link':
            f.seek(last_pos)

        reshares = []
        post_memes = []

        if not ignoring:

            # Create posts, postmemes, and reshares for the edges not visited before.
            for i in range(len(uid_list) - 1):
                resh_dt = retweet_time - timedelta(microseconds=(len(uid_list) - i - 2) * 10)
                src, dst = uid_list[i], uid_list[i + 1]

                #if (str(src), str(dst)) not in resh_pairs:
                resh_post = posts_map[str(src)]
                resh_post_id = resh_post['_id']

                if i == len(uid_list) - 1:
                    post_id = retweet_pid
                    mongodb.posts.insert_one({'_id': post_id, 'author_id': dst, 'datetime': resh_dt})
                else:
                    post_id = mongodb.posts.insert_one({'author_id': dst, 'datetime': resh_dt}).inserted_id

                resh = {'post_id': post_id, 'reshared_post_id': resh_post_id, 'datetime': resh_dt,
                        'user_id': dst, 'ref_user_id': src, 'ref_datetime': resh_post['datetime']}
                reshares.append(resh)
                post_memes.append({'post_id': post_id, 'meme_id': meme_id, 'datetime': resh_dt})
                posts_map[str(dst)] = {'_id': post_id, 'datetime': resh_dt}
                #resh_pairs.add((str(src), str(dst)))

        return reshares, post_memes


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

    def create_indexes(self):
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


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
