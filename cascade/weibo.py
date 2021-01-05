import logging
import time
import os

from bson.objectid import ObjectId
from datetime import timedelta
from networkx import DiGraph, write_adjlist
from pymongo.errors import BulkWriteError
from pymongo.operations import UpdateOne

from memm.db import DBManager
import settings
from utils.time_utils import str_to_datetime, time_measure

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('weibo')
logger.setLevel(settings.LOG_LEVEL)


def create_roots(path):
    """
    Create memes and their root posts reading the file Root_Content.txt .
    :param path: path of the file Root_Content.txt
    """
    posts = []
    post_memes = []
    memes_map = {}
    i = 0
    db = DBManager().db

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
                meme_id = db.memes.insert_one({'text': content}).inserted_id
                post_memes.append({'post_id': post_id_obj, 'meme_id': meme_id})
                memes_map[post_id] = meme_id
                post_id = None

                i += 1
                if i % 10000 == 0:
                    logger.info('%d posts read' % i)
                if i % 100000 == 0:
                    db.posts.insert_many(posts)
                    db.postmemes.insert_many(post_memes)
                    logger.info('%d memes and their original posts created' % i)
                    posts = []
                    post_memes = []

            line = f.readline()

    if posts:
        db.posts.insert_many(posts)
        db.postmemes.insert_many(post_memes)
        logger.info('%d memes and their original posts created' % i)

    return memes_map


def to_user_id(number):
    """
    Convert number to 24-length valid db user id.
    :param number: number string
    :return: 24-length valid db user id (instance of ObjectId)
    """
    return ObjectId('{:024d}'.format(int(number)))


def create_users(paths):
    """
    Create users reading the files user_profile1.txt and user_profile2.txt .
    :param paths: list of path of the files
    :returns map of usernames to user ids
    """
    users = []
    db = DBManager().db
    user_ids = {u['_id'] for u in db.users.find({}, ['_id'])}
    users_map = {u['username']: u['_id'] for u in db.users.find({}, ['_id', 'username'])}
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

                    user_id = to_user_id(line)
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
                            db.users.insert_many(users)
                            logger.info('%d users created' % i)
                            users = []

                    for _ in range(6):
                        line = f.readline()

                except BulkWriteError as e:
                    print(e.details)
                    raise

        if users:
            db.users.insert_many(users)
            logger.info('%d users created' % i)
            users = []

    return users_map, user_ids

# @profile
def create_retweets(path, start_index, users_map, user_ids, memes_map):
    i = 0
    t0 = time.time()
    db = DBManager().db
    memes_count = db.memes.count()
    post_memes = []
    reshares = []

    # if 'start_index' is specified, ignore lower indexes.
    ignoring = False
    if start_index:
        ignoring = True

    with open(path, encoding='GB18030', errors='replace') as f:

        while True:
            i += 1
            resh_list, pm_list = read_one_meme_reshares(f, users_map, user_ids, memes_map, ignoring)
            if resh_list is None and pm_list is None:
                break
            reshares.extend(resh_list)
            post_memes.extend(pm_list)

            if (not ignoring or i == start_index) and (len(post_memes) > 10000 or len(reshares) > 10000):
                logger.info(
                    'saving %d post memes and %d reshares ...' % (len(post_memes), len(reshares)))
                if post_memes or reshares:
                    db.postmemes.insert_many(post_memes)
                    db.reshares.insert_many(reshares)
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
    db.postmemes.insert_many(post_memes)
    db.reshares.insert_many(reshares)


def read_one_meme_reshares(f, users_map, user_ids, memes_map, ignoring=False):
    # Read root post data.
    db = DBManager().db
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
        db.users.insert_one({'_id': original_uid, 'username': None})
        user_ids.append(original_uid)

    if not ignoring:
        db.posts.update_one({'_id': original_pid},
                                 {'$set': {'datetime': original_time, 'author_id': original_uid}})
        db.postmemes.update_one({'post_id': original_pid}, {'$set': {'datetime': original_time}})
    meme_id = memes_map[str(original_pid)]

    # Read retweets number.
    line = f.readline().strip()
    retweet_num = int(line)

    meme_reshares = []
    meme_postmemes = []
    # resh_pairs = set()
    posts_map = {str(original_uid): {'_id': original_pid, 'datetime': original_time}}

    for i in range(retweet_num):
        reshares, post_memes = read_one_reshare_seq(f, meme_id, original_uid, users_map, posts_map, ignoring)
        meme_reshares.extend(reshares)
        meme_postmemes.extend(post_memes)

    return meme_reshares, meme_postmemes


def read_one_reshare_seq(f, meme_id, original_uid, users_map, posts_map, ignoring=False):
    db = DBManager().db

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
                user_id = db.users.insert_one({'username': uname}).inserted_id
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

            # if (str(src), str(dst)) not in resh_pairs:
            resh_post = posts_map[str(src)]
            resh_post_id = resh_post['_id']

            if i == len(uid_list) - 1:
                post_id = retweet_pid
                db.posts.insert_one({'_id': post_id, 'author_id': dst, 'datetime': resh_dt})
            else:
                post_id = db.posts.insert_one({'author_id': dst, 'datetime': resh_dt}).inserted_id

            resh = {'post_id': post_id, 'reshared_post_id': resh_post_id, 'datetime': resh_dt,
                    'user_id': dst, 'ref_user_id': src, 'ref_datetime': resh_post['datetime']}
            reshares.append(resh)
            post_memes.append({'post_id': post_id, 'meme_id': meme_id, 'datetime': resh_dt})
            posts_map[str(dst)] = {'_id': post_id, 'datetime': resh_dt}
            # resh_pairs.add((str(src), str(dst)))

    return reshares, post_memes


def calc_memes_values():
    db = DBManager().db
    count = db.memes.count()
    save_step = 10 ** 6

    logger.info('query of meme counts ...')
    meme_counts = db.postmemes.aggregate([{'$group': {'_id': '$meme_id', 'count': {'$sum': 1}}}],
                                              allowDiskUse=True)

    logger.info('saving ...')
    operations = []
    i = 0
    for doc in meme_counts:
        operations.append(UpdateOne({'_id': doc['_id']}, {'$set': {'count': doc['count']}}))
        i += 1
        if i % save_step == 0:
            db.memes.bulk_write(operations)
            operations = []
            logger.info('%d%% done', i * 100 / count)
    db.memes.bulk_write(operations)

    logger.info('query of first times ...')
    first_times = db.postmemes.aggregate([{'$group': {'_id': '$meme_id', 'first': {'$min': '$datetime'}}}],
                                              allowDiskUse=True)

    logger.info('saving ...')
    operations = []
    i = 0
    for doc in first_times:
        operations.append(UpdateOne({'_id': doc['_id']}, {'$set': {'first_time': doc['first']}}))
        i += 1
        if i % save_step == 0:
            db.memes.bulk_write(operations)
            operations = []
            logger.info('%d%% done', i * 100 / count)
    db.memes.bulk_write(operations)

    logger.info('query of last times ...')
    last_times = db.postmemes.aggregate([{'$group': {'_id': '$meme_id', 'last': {'$max': '$datetime'}}}],
                                             allowDiskUse=True)

    logger.info('saving ...')
    operations = []
    i = 0
    for doc in last_times:
        operations.append(UpdateOne({'_id': doc['_id']}, {'$set': {'last_time': doc['last']}}))
        i += 1
        if i % save_step == 0:
            db.memes.bulk_write(operations)
            operations = []
            logger.info('%d%% done', i * 100 / count)
    db.memes.bulk_write(operations)


def read_uidlist(uidlist_file):
    logger.info('reading uidlist ...')
    uid_list = []
    with open(uidlist_file, encoding='utf-8', errors='ignore') as f:
        line = f.readline()
        while line:
            uid = line.strip()
            if uid:
                uid_list.append(to_user_id(uid))
            line = f.readline()
    return uid_list


@time_measure()
def extract_relations(relations_file, uidlist_file, user_ids=None):
    db = DBManager().db

    uid_list = read_uidlist(uidlist_file)
    uid_list_map = {uid_list[i]: i for i in range(len(uid_list))}

    user_ids_indexes = None
    if user_ids:
        user_ids_indexes = {uid_list_map[uid] for uid in user_ids if uid in uid_list_map}

    logger.info('reading relationships...')
    i = 0
    edges = []

    with open(relations_file, encoding='utf-8', errors='ignore') as f:
        f.readline()
        line = f.readline()

        while line:
            line = line.strip().split()
            u1_i = int(line[0])
            u1 = uid_list[u1_i]
            n = int(line[1])
            for j in range(n):
                u2_i = int(line[2 + j * 2])
                u2 = uid_list[u2_i]
                rel_type = line[3 + j * 2]
                if user_ids_indexes is None or u1_i in user_ids_indexes or u2_i in user_ids_indexes:
                    edges.append({'parent': u2, 'child': u1})
                    if rel_type == '1':
                        edges.append({'parent': u1, 'child': u2})

            i += 1
            if i % 10000 == 0:
                logger.info('%d lines read. saving ...' % i)
                db.relations.insert_many(edges)
                logger.info('%d new relations saved' % len(edges))
                edges = []

            line = f.readline()

    if edges:
        db.relations.insert_many(edges)
        logger.info('%d edges saved' % len(edges))

