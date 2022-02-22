import argparse
import json
import logging
import os
import pprint

from bson import ObjectId
from pymongo import IndexModel

from cascade.models import CascadeTree
from db.managers import DBManager
from settings import logger, LOG_LEVEL, BASE_PATH
from utils.time_utils import time_measure, str_to_datetime


def read_tweet(f):
    username = f.readline()
    if not username:
        return None
    else:
        username = username.strip()
    tweet_id = f.readline().strip()
    date = f.readline().strip().replace('+0000', '').replace('UTC', '')
    date = str_to_datetime(date, '%a %b %d %H:%M:%S  %Y', zone='UTC')
    f.readline()
    retweet_from = f.readline().strip()
    # if retweet_from == '-1':
    #     retweet_from = None
    reply_to = f.readline().strip()
    if reply_to != '-1':
        parts = reply_to.split()
        if len(parts) == 2:
            reply_to_user, reply_to_tweet = parts
        else:
            reply_to_user, reply_to_tweet = None, parts[-1]
    else:
        reply_to_user, reply_to_tweet = None, None
    content = f.readline().strip()
    links_num = int(f.readline().strip())
    # links = [f.readline().strip().split() for _ in range(links_num)]
    for _ in range(links_num):
        f.readline()
    f.readline()

    tweet = {
        'username': username,
        'tweet_id': tweet_id,
        'date': date,
    }
    if reply_to_tweet:
        tweet['reply_to_id'] = reply_to_tweet
    return tweet


def insert_tweets(dataset_path, db):
    dirs = [
        os.path.join(dataset_path, '2010_10_14'),
        os.path.join(dataset_path, '2011_01_10')
    ]
    tweets = []
    tweet_count = 0
    file_count = 0

    for directory in dirs:
        i = 0

        for fname in os.listdir(directory):
            if fname[-3:] == 'txt' and fname != 'count.txt':
                path = os.path.join(directory, fname)
                logger.info('reading file %s ...', path)

                with open(path, encoding='ISO-8859-1') as f:
                    while True:
                        tweet = read_tweet(f)
                        if tweet is None:
                            break
                        tweets.append(tweet)

                        tweet_count += 1
                        if tweet_count % 100000 == 0:
                            logger.info('saving tweets ...')
                            db.tweets.insert_many(tweets)
                            tweets = []
                            logger.info('%d tweets read', tweet_count)
                            logger.info('%d files completed', file_count)

                file_count += 1
                i += 1

    if tweets:
        logger.info('saving tweets ...')
        db.tweets.insert_many(tweets)
        logger.info('%d tweets read', tweet_count)
        logger.info('%d files read', file_count)

    logger.info('creating indexes ...')
    db.tweets.create_indexes([IndexModel('date'), IndexModel('tweet_id')])
    logger.info('done')


def log_stat(cascades, tweet_count, render_in_level=logging.DEBUG):
    logger.info('%d tweets done', tweet_count)
    logger.info('%d cascades extracted', len(cascades))

    depths = {}
    for cas in cascades:
        depth = cas.depth
        depths.setdefault(depth, 0)
        depths[depth] += 1
    logger.info('cascades grouped by depth: %s', depths)

    if LOG_LEVEL <= render_in_level:
        logger.log(render_in_level, 'cascades:')
        for cas in cascades:
            logger.log(render_in_level, '\n%s\n', cas.render(digest=True))


def save_cascades_data(cascades, cascade_user_ids, tweet_to_cascade):
    logger.info('saving cascades data to file ...')
    data_dir = os.path.join(BASE_PATH, 'data', 'twitter')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    cascades_path = os.path.join(data_dir, 'cascades.json')
    user_ids_path = os.path.join(data_dir, 'cascade_user_ids.json')
    tweet_to_cascade_path = os.path.join(data_dir, 'tweet_to_cascade.json')
    with open(cascades_path, 'w') as f:
        json.dump([tree.to_json() for tree in cascades], f, indent=1)
    with open(user_ids_path, 'w') as f:
        json.dump([[str(uid) for uid in user_ids] for user_ids in cascade_user_ids], f, indent=1)
    with open(tweet_to_cascade_path, 'w') as f:
        json.dump(tweet_to_cascade, f, indent=1)
    logger.info('done')


def load_cascades_data():
    logger.info('loading cascades data from file ...')
    data_dir = os.path.join(BASE_PATH, 'data', 'twitter')
    cascades_path = os.path.join(data_dir, 'cascades.json')
    user_ids_path = os.path.join(data_dir, 'cascade_user_ids.json')
    tweet_to_cascade_path = os.path.join(data_dir, 'tweet_to_cascade.json')
    with open(cascades_path) as f:
        cascades = [CascadeTree().from_json(tree) for tree in json.load(f)]
    with open(user_ids_path) as f:
        cascade_user_ids = [{ObjectId(uid) for uid in user_ids} for user_ids in json.load(f)]
    with open(tweet_to_cascade_path) as f:
        tweet_to_cascade = json.load(f)
    logger.info('done')
    return cascades, cascade_user_ids, tweet_to_cascade


def extract_cascades(db, users_map):
    logger.info('iterating on tweets to extract cascades ...')
    i = 0
    # i = 150000

    tweets = db.tweets.find(no_cursor_timeout=True).sort('date')
    # if i == 0:
    cascades = []  # list of instances of CascadeTree of cascades
    cascade_user_ids = []  # list of usernames set related to cascades. cascade_usernames[i] is related to cascades[i]
    tweet_to_cascade = {}  # dictionary of tweet ids to cascade index
    # if i > 0:
    #     cascades, cascade_user_ids, tweet_to_cascade = load_cascades_data()
    #     logger.info('skipping %d tweets', i)
    #     tweets = tweets.skip(i)

    for tweet in tweets:
        user_id = users_map[tweet['username']]

        if 'reply_to_id' in tweet:
            parent = db.tweets.find_one({'tweet_id': tweet['reply_to_id']})

            if parent and parent['username'] != tweet['username']:
                parent_id = users_map[parent['username']]

                if parent['tweet_id'] in tweet_to_cascade:
                    cas_index = tweet_to_cascade[parent['tweet_id']]
                    tree = cascades[cas_index]
                    cas_user_ids = cascade_user_ids[cas_index]
                    if user_id not in cas_user_ids:
                        cas_user_ids.add(user_id)
                        tree.add_node(user_id, act_time=tweet['date'], parent_id=parent_id)
                        tweet_to_cascade[tweet['tweet_id']] = cas_index
                else:
                    tree = CascadeTree()
                    logger.debug('date = %s', tweet['date'])
                    logger.debug('type of date = %s', type(tweet['date']))
                    tree.add_node(parent_id, act_time=parent['date'])
                    tree.add_node(user_id, act_time=tweet['date'], parent_id=parent_id)
                    cascades.append(tree)
                    cas_user_ids = {parent_id, user_id}
                    cascade_user_ids.append(cas_user_ids)
                    tweet_to_cascade[tweet['tweet_id']] = len(cascades) - 1
                    tweet_to_cascade[parent['tweet_id']] = len(cascades) - 1

        i += 1
        if i % 10000 == 0:
            log_stat(cascades, i)

        # Save the results to continue after unwanted stop.
        if i % 10 ** 6 == 0:
            save_cascades_data(cascades, cascade_user_ids, tweet_to_cascade)

    log_stat(cascades, i, logging.INFO)

    return cascades


def insert_users(db):
    logger.info('extracting usernames ...')
    results = db.tweets.aggregate([{'$group': {'_id': '$username'}}])
    users = [{'username': doc['_id']} for doc in results]
    logger.info('inserting users ...')
    db.users.insert_many(users)
    logger.info('done')
    users_map = {user['username']: user['_id'] for user in users}
    return users_map


def insert_data(cascades, db):
    post_cascades = []
    reshares = []
    i = 0
    logger.info('iterating on trees to insert cascades, posts, posts_cascades, and reshares ...')

    for tree in cascades:
        # Insert the posts of the tree.
        posts = [{
            'author_id': node.user_id,
            'datetime': node.datetime
        } for node in tree.nodes()]
        db.posts.insert_many(posts)
        posts_map = {p['author_id']: p for p in posts}

        # Insert the cascade.
        datetimes = [node.datetime for node in tree.nodes()]
        first_time, last_time = min(datetimes), max(datetimes)
        cascade_id = db.cascades.insert_one({
            'size': len(posts),
            'count': len(posts),
            'depth': tree.depth,
            'first_time': first_time,
            'last_time': last_time
        }).inserted_id

        # Create the post_cascades the tree and add them to the list.
        cas_post_cascades = [{'post_id': post['_id'],
                              'cascade_id': cascade_id,
                              'author_id': post['author_id'],
                              'datetime': post['datetime']} for post in posts]
        post_cascades.extend(cas_post_cascades)

        # Create the reshares of the tree and add them to the list.
        cas_reshares = [{
            'reshared_post_id': posts_map[edge[0]]['_id'],
            'post_id': posts_map[edge[1]]['_id'],
            'ref_user_id': edge[0],
            'user_id': edge[1],
            'ref_datetime': posts_map[edge[0]]['datetime'],
            'datetime': posts_map[edge[1]]['datetime']
        } for edge in tree.edges()]
        reshares.extend(cas_reshares)

        i += 1
        if i % 1000 == 0:
            logger.info('%d trees done', i)

        # Insert the reshares and post_cascades if they reached to a sufficient number.
        if i % 100000 == 0:
            logger.info('inserting reshares and post_cascades ...')
            db.reshares.insert_many(reshares)
            db.postcascades.insert_many(post_cascades)
            logger.info('done')
            reshares = []
            post_cascades = []

    if reshares or post_cascades:
        logger.info('inserting reshares and post_cascades ...')
        db.reshares.insert_many(reshares)
        db.postcascades.insert_many(post_cascades)
        logger.info('done')


@time_measure()
def main(args):
    manager = DBManager(args.db)

    # Drop the database.
    manager.client.drop_database(args.db)
    db = manager.db

    insert_tweets(args.path, db)
    users_map = insert_users(db)
    # users_map = {user['username']: user['_id'] for user in db.users.find()
    cascades = extract_cascades(db, users_map)
    del users_map
    insert_data(cascades, db)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Read Enron dataset')
    parser.add_argument("-p", "--path", required=True, help="dataset directory path")
    parser.add_argument('-d', '--db', required=True, help="db name in which the documents must be inserted")
    args = parser.parse_args()
    main(args)
