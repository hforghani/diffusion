import argparse
import json
import os
import pprint
import numpy as np
from bson import ObjectId
from networkx import DiGraph

import sampledata
from cascade.models import CascadeTree, ActSequence, Project, ParamTypes
from db.managers import DBManager
from settings import logger
from utils.time_utils import time_measure, str_to_datetime


def read_users(db, path):
    usernames = set()
    for file_name in os.listdir(path):
        logger.debug('reading file %s ...', file_name)
        with open(os.path.join(path, file_name)) as f:
            data = json.load(f)
        for post_data in data:
            usernames.add(post_data['user_id'])
            if 'repost' in post_data and 'username' in post_data['repost']:
                usernames.add(post_data['repost']['username'])
    logger.debug('inserting %d users ...', len(usernames))
    db.users.insert_many([{'username': uname} for uname in usernames])


def read_posts(db, path):
    users_map = {user['username']: user['_id'] for user in db.users.find()}
    posts_map = {}
    reshares = []

    for file_name in os.listdir(path):
        # for file_name in ['1.json']:  # TODO
        logger.debug('reading file %s ...', file_name)
        with open(os.path.join(path, file_name)) as f:
            data = json.load(f)

        for post_data in data:
            user_id = users_map[post_data['user_id']]
            dt = str_to_datetime(post_data['timestamp'], '%Y-%m-%dT%H:%M', zone='UTC')
            res = db.posts.insert_one({
                'author_id': user_id,
                'datetime': dt
            })
            posts_map.setdefault(user_id, {})
            posts_map[user_id][dt] = post_id = res.inserted_id

            if 'repost' in post_data and 'timestamp' in post_data['repost'] and 'username' in post_data['repost']:
                ref_dt = str_to_datetime(post_data['repost']['timestamp'], '%Y-%m-%dT%H:%M', zone='UTC')
                ref_user_id = users_map[post_data['repost']['username']]
                reshares.append({
                    'reshared_post_id': None,  # Will be set later.
                    'post_id': post_id,
                    'ref_user_id': ref_user_id,
                    'user_id': user_id,
                    'ref_datetime': ref_dt,
                    'datetime': dt
                })

    logger.debug('filling reshared post ids ...')
    for resh in reshares:
        ref_uid, ref_dt = resh['ref_user_id'], resh['ref_datetime']
        if ref_uid not in posts_map or ref_dt not in posts_map[ref_uid]:
            res = db.posts.insert_one({
                'author_id': ref_uid,
                'datetime': ref_dt
            })
            posts_map.setdefault(ref_uid, {})
            posts_map[ref_uid][ref_dt] = reshared_post_id = res.inserted_id
        else:
            reshared_post_id = posts_map[ref_uid][ref_dt]
            # if ref_uid == ObjectId("62eba7413cf761035f1f94a1"):
            #     logger.debug('%s, %s found', ref_uid, ref_dt)
        resh['reshared_post_id'] = reshared_post_id

    logger.debug('inserting %d reshares ...', len(reshares))
    db.reshares.insert_many(reshares)
    logger.debug('creating index ...')
    db.reshares.create_index('datetime')


def extract_trees(db):
    # TODO: Fix the bug: all trees are of size 2.
    trees = []
    posts_data = {}

    for resh in db.reshares.find().sort('datetime'):
        uid, ref_uid = resh['user_id'], resh['ref_user_id']
        if uid == ref_uid:
            continue
        pid, resh_pid = resh['post_id'], resh['reshared_post_id']
        dt, ref_dt = resh['datetime'], resh['ref_datetime']

        # if resh_pid == ObjectId("62ebb7bc3cf761035f6707cd"):
        #     logger.debug('pid = %s in posts_data = %s', pid, pid in posts_data)
        #     logger.debug('resh_pid = %s in posts_data = %s', resh_pid, resh_pid in posts_data)

        if pid not in posts_data:
            if resh_pid not in posts_data:
                tree = CascadeTree()
                tree.add_node(ref_uid, ref_dt)
                tree.add_node(uid, dt, ref_uid)
                trees.append(tree)
                ind = len(trees) - 1
                posts_data[resh_pid] = {'cascade_ind': ind, 'author_id': ref_uid, 'datetime': ref_dt}
            else:
                ind = posts_data[resh_pid]['cascade_ind']
                tree = trees[ind]
                if tree.get_node(uid) is None:
                    tree.add_node(uid, dt, ref_uid)

            posts_data[pid] = {'cascade_ind': ind, 'author_id': uid, 'datetime': dt}

            # if resh_pid == ObjectId("62ebb7bc3cf761035f6707cd"):
            #     logger.debug('ind = %s', ind)
            #     logger.debug('tree.size() = %s', tree.size())
            #     logger.debug('posts_data[pid] = %s', posts_data[pid])

    return trees, posts_data


def insert_cascades(db, trees, posts_data):
    cascade_ids = []
    for tree in trees:
        size = tree.size()
        cid = db.cascades.insert_one({
            'size': size,
            'count': size,
            'depth': tree.depth,
            'first_time': min(node.datetime for node in tree.roots),
            'last_time': max(node.datetime for node in tree.nodes())
        }).inserted_id
        cascade_ids.append(cid)

    logger.debug('inserting %d postcascasdes ...', len(posts_data))
    post_cascades = [{'post_id': pid,
                      'cascade_id': cascade_ids[data['cascade_ind']],
                      'author_id': data['author_id'],
                      'datetime': data['datetime']}
                     for pid, data in posts_data.items()]
    db.postcascades.insert_many(post_cascades)

    return cascade_ids


def extract_act_sequences(trees):
    sequences = {}
    for cid, tree in trees.items():
        nodes = tree.nodes()
        indexes = np.argsort(np.array([n.datetime for n in nodes]))
        user_ids = [nodes[i].user_id for i in indexes]
        first_time = nodes[indexes[0]].datetime
        times = [(nodes[i].datetime - first_time).total_seconds() / (3600.0 * 24 * 30) for i in
                 indexes]  # number of months
        sequences[cid] = ActSequence(user_ids, times, times[-1])
    return sequences


def save_graph(db, project):
    graph = DiGraph()
    edges = {(resh['ref_user_id'], resh['user_id']) for resh in db.reshares.find()}
    cid_strs = [str(c['_id']) for c in db.cascades.find({}, ['_id'])]
    graph.add_edges_from(edges)
    graph_info_fname = 'graph_info'
    fname = 'graph1'
    project.save_param(graph, fname, ParamTypes.GRAPH)
    graph_info = {fname: cid_strs}
    project.save_param(graph_info, graph_info_fname, ParamTypes.JSON)


@time_measure()
def main(db_name, path):
    manager = DBManager(db_name)

    manager.client.drop_database(db_name)  # Drop the database.

    db = manager.db
    project_name = 'weibocovid-all'
    project = Project(project_name, 'weibocovid')

    logger.info('reading users ...')
    read_users(db, path)
    logger.info('reading posts and reshares ...')
    read_posts(db, path)

    logger.info('extracting trees ...')
    trees, posts_data = extract_trees(db)
    logger.info('inserting cascades ...')
    cascade_ids = insert_cascades(db, trees, posts_data)
    del posts_data
    trees = dict(zip(cascade_ids, trees))
    project.save_trees(trees)

    # trees = project.load_trees()

    logger.info('extracting activation sequences ...')
    sequences = extract_act_sequences(trees)
    del trees
    logger.info('saving activation sequences ...')
    project.save_act_sequences(sequences)
    del sequences
    logger.info('saving graph ...')
    save_graph(db, project)

    logger.info('sampling training and test sets ...')
    sampledata.Command().sample_data(db_name, project_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Read Weibo Covid dataset')
    parser.add_argument("-p", "--path", required=True, help="dataset directory path")
    parser.add_argument('-d', '--db', required=True, help="db name in which the documents must be inserted")
    args = parser.parse_args()
    main(args.db, args.path)
