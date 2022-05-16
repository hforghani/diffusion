import logging
import math
import pprint
import sys
from multiprocessing import Pool
from typing import Dict, List, Set
import argparse
import json
import os
from bson import ObjectId
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.preprocessing import normalize
import numpy as np
from matplotlib import pyplot as plt
import random

from sknetwork.hierarchy import Paris

sys.path.append('.')

import settings
from db.managers import DBManager
from settings import logger, BASE_PATH
from cascade.models import Project
from utils.time_utils import Timer


def get_users(cascade_id: str, db_name) -> Set[str]:
    db = DBManager(db_name).db
    users = db.postcascades.find({'cascade_id': ObjectId(cascade_id)}, {'author_id': 1, '_id': 0})
    return {str(u['author_id']) for u in users}


def calc_jaccards(index_pairs, users, cascade_ids):
    logger.debug('calculating %d jaccard coefficients', len(index_pairs))
    logger.debug('len(users) = %d', len(users))
    jaccards = {}
    for i, j in index_pairs:
        inter = len(users[cascade_ids[i]] & users[cascade_ids[j]])
        if inter == 0:
            val = 0
        else:
            val = inter / len(users[cascade_ids[i]] | users[cascade_ids[j]])
        jaccards[(i, j)] = val
    logger.debug('done')
    return jaccards


def get_jaccard_mat_mp(cascades: List[str], users: Dict[str, set]):
    count = len(cascades)
    results = []
    process_count = settings.PROCESS_COUNT
    n = int(math.floor((math.sqrt(1 + 8 * process_count) - 1) / 2))
    process_count = min(process_count, int(n * (n + 1) / 2))
    pool = Pool(processes=process_count)
    logger.debug('n = %d', n)
    step = int(math.ceil(count / n))
    logger.debug('step = %d', step)

    for i in range(0, count, step):
        for j in range(i, count, step):
            logger.debug('starting process for index %d to %d with index %d to %d', i, min(i + step, count), j,
                         min(j + step, count))
            cur_pairs = [(r, c) for r in range(i, min(i + step, count)) for c in range(j, min(j + step, count)) if
                         r < c]
            cur_cascades = {cascades[r] for r in range(i, min(i + step, count))}
            cur_cascades.update(cascades[c] for c in range(j, min(j + step, count)))
            cur_users = {cid: users[cid] for cid in cur_cascades}
            res = pool.apply_async(calc_jaccards, (cur_pairs, cur_users, cascades))
            results.append(res)

        for k in range(i, min(i + step, count)):
            del users[cascades[k]]  # to free RAM

    pool.close()
    pool.join()

    mat = np.zeros((count, count))
    for res in results:
        jaccard_values = res.get()
        for i, j in jaccard_values:
            mat[i, j] = jaccard_values[(i, j)]

    mat += mat.transpose() + np.eye(count)
    return mat


def get_jaccard_mat(cascades: List[str], users: Dict[str, set]):
    logger.info('calculating jaccard coefficients ...')
    count = len(cascades)
    mat = np.zeros((count, count), dtype=np.float32)

    pairs_count = count * (count - 1) / 2
    counter = 0
    for i in range(count - 1):
        for j in range(i + 1, count):
            mat[i, j] = len(users[cascades[i]] & users[cascades[j]]) / len(users[cascades[i]] | users[cascades[j]])
            counter += 1
            if counter % 1000 == 0:
                logger.debug('%d%% done', counter / pairs_count * 100)

    mat += mat.transpose() + np.eye(count)
    return mat


def heat_map(mat, file_name):
    fig, ax = plt.subplots()
    # im = ax.imshow(mat)
    # plt.show()
    plt.imsave(file_name, mat)


def hierarchical(mat, clust_num):
    paris = Paris()
    dendrogram = paris.fit_transform(mat)
    clusters = {i: [i] for i in range(mat.shape[0])}
    new_clust_index = len(clusters)
    # logger.debug('len(clusters) = \n%s', len(clusters))
    # logger.debug('dendrogram = \n%s', dendrogram)

    for i, j, dist, size in dendrogram:
        if len(clusters) <= clust_num:
            break
        i, j = int(i), int(j)
        # logger.debug('merging %d and %d into %d', i, j, new_clust_index)
        clusters[new_clust_index] = clusters[i] + clusters[j]
        new_clust_index += 1
        del clusters[i]
        del clusters[j]

    keys = list(clusters.keys())
    key_map = {keys[i]: i for i in range(len(keys))}
    clusters = {key_map[key]: value for key, value in clusters.items()}
    labels = np.zeros(mat.shape[0], dtype=int)
    for label, indexes in clusters.items():
        labels[indexes] = label

    logger.debug('labels = \n%s', labels)

    return labels


def cluster_mat(mat, clust_num, method):
    if method == 'spectral':
        clustering = SpectralClustering(n_clusters=clust_num,
                                        assign_labels="discretize",
                                        random_state=0).fit(mat)
        return clustering.labels_
    elif method == 'dbscan':
        clustering = DBSCAN(eps=0.001, min_samples=5, metric='precomputed', n_jobs=-1).fit(mat)
        return clustering.labels_
    elif method == 'hierarchy':
        return hierarchical(mat, clust_num)


def load_or_extract_users(cas_ids: List[str], db_name: str) -> Dict[str, set]:
    fname = os.path.join(BASE_PATH, 'data', f'{db_name}_users.json')
    try:
        with open(fname) as f:
            logger.info('loading users lists ...')
            users = json.load(f)
    except (FileNotFoundError, ValueError):
        users = {}

    res_users = {}
    i = 1
    changed = False
    logger.info('querying users of %d cascades', len(cas_ids))
    for cas_id in cas_ids:
        if cas_id not in users:
            logger.debug('querying users of cascade {}'.format(i))
            users_set = get_users(cas_id, db_name)
            users[cas_id] = list(users_set)
            res_users[cas_id] = users_set
            changed = True
        else:
            res_users[cas_id] = set(users[cas_id])
        i += 1
        if i % 1000 == 0:
            logger.info('%d cascades done', i)

    if changed:
        logger.info('saving users lists ...')
        with open(fname, 'w') as f:
            json.dump(users, f)

    return res_users


def calc_error(mat, clusters):
    count = mat.shape[0]
    squares_mat = np.zeros((count, count))
    begin = 0
    for label in sorted(clusters.keys()):
        clust_size = len(clusters[label])
        squares_mat[begin: begin + clust_size, begin: begin + clust_size] = 1
        begin += clust_size
    error = np.linalg.norm(mat - squares_mat)
    return error


def save_clusters(clust_num, db_name, min_size, max_size, depth, method, file_name):
    base_file_name = os.path.join(BASE_PATH, 'data', 'clusters',
                                  f'{db_name}-{min_size}to{max_size}-{depth}')
    mat_file_name = base_file_name + '-mat.npy'
    cascades_file_name = base_file_name + '-cascades.json'

    if os.path.exists(mat_file_name) and os.path.exists(cascades_file_name):
        mat = np.load(mat_file_name)
        with open(cascades_file_name) as f:
            cascades = json.load(f)
        logger.info('matrix loaded')

    else:
        # Extract the cascades.
        logger.info('loading cascades ...')
        db = DBManager(db_name).db
        query = {}
        if min_size is not None:
            query['size'] = {'$gte': min_size}
        if max_size is not None:
            query.setdefault('size', {})
            query['size']['$lte'] = max_size
        if depth is not None:
            query['depth'] = {'$gte': depth}
        res_cascades = list(db.cascades.find(query, ['_id']))
        random.shuffle(res_cascades)  # To balance the processes in multi-processing
        cascades = [str(m['_id']) for m in res_cascades]
        logger.info('%d cascades found', len(cascades))

        # Extract user sets of top cascades
        users = load_or_extract_users(cascades, db_name)

        # Calculate the Jaccard matrix.
        try:
            logger.info('creating the similarity matrix using multiple processes ...')
            mat = get_jaccard_mat_mp(cascades, users)
        except MemoryError:
            logger.info('creating the similarity matrix using single process ...')
            mat = get_jaccard_mat(cascades, users)

        # Normalize the matrix.
        mat = np.reshape(mat - np.eye(len(cascades)), (1, mat.size))
        mat = normalize(mat, norm='max')
        mat = np.reshape(mat, (len(cascades), len(cascades)))
        mat += np.eye(len(cascades))

        with open(cascades_file_name, 'w') as f:
            json.dump(cascades, f)
        np.save(mat_file_name, mat)

    # Cluster the cascades.
    logger.info('clustering the cascades ...')
    labels = cluster_mat(mat, clust_num, method)

    # Create the ordered indexes of cascades.
    ordered_ind = np.array([], dtype=np.int64)
    uni_val = np.unique(labels)
    clusters = {}

    # Add file handler to the logger.
    file_handler = logging.FileHandler(file_name + '.out', 'w', 'utf-8')
    file_handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
    logger.addHandler(file_handler)

    logger.info('%d cluster(s) found', uni_val.size)
    for val in uni_val:
        indexes = np.nonzero(labels == val)[0]
        indexes = indexes.astype(np.int64)
        clusters[val] = [cascades[i] for i in indexes]
        ordered_ind = np.concatenate((ordered_ind, indexes))
        mean = np.mean(mat[indexes, :][:, indexes])
        # logger.debug('mat[indexes, indexes] = %s', mat[indexes, :][:, indexes])
        logger.info('cluster %d with %d cascades and mean Jaccard value of %f', val, indexes.size, mean)

    # Save the clusters into the file.
    with open(file_name + '.json', 'w') as f:
        json.dump({str(key): clust for key, clust in clusters.items()}, f, indent=4)

    new_mat = mat[:, ordered_ind]
    new_mat = new_mat[ordered_ind, :]

    # Calculate the clustering error.
    error = calc_error(new_mat, clusters)
    logger.info('error = %f', error)

    heat_map(new_mat, file_name + '.png')

    return clusters


def save_project(project_name, db_name, cas_ids):
    cas_ids_copy = cas_ids.copy()
    random.shuffle(cas_ids_copy)
    test_ratio = 0.3
    test_num = round(test_ratio * len(cas_ids_copy))
    test_set, train_set = cas_ids_copy[:test_num], cas_ids_copy[test_num:]
    project = Project(project_name, db_name)
    project.save_sets(train_set, test_set)


def main():
    parser = argparse.ArgumentParser('Cluster the cascades based on their common users')
    parser.add_argument('-d', '--db', required=True, help='db name')
    parser.add_argument('-c', '--clusters', type=int, default=5, help='number of clusters')
    parser.add_argument('-m', '--min', type=int, help='minimum cascade size')
    parser.add_argument('-M', '--max', type=int, help='maximum cascade size')
    parser.add_argument('--method', choices=['spectral', 'dbscan', 'hierarchy'], help='clustering method')
    parser.add_argument('-D', '--depth', type=int, help='minimum depth')
    parser.add_argument('-i', '--create_by_index', type=int, nargs='+',
                        help='Create a project using cascade ids in cluster index given')
    parser.add_argument('-p', '--project', help='the project name to create if --create_by_index is given')

    args = parser.parse_args()
    method = args.method
    file_name = os.path.join(BASE_PATH, 'data', 'clusters',
                             f'{args.db}-{args.min}to{args.max}-{args.depth}-{method}-{args.clusters}')
    clusters_file_name = file_name + '.json'

    if os.path.exists(clusters_file_name):
        with open(clusters_file_name) as f:
            clusters = {int(key): value for key, value in json.load(f).items()}
    else:
        clusters = save_clusters(args.clusters, args.db, args.min, args.max, args.depth, method, file_name)

    if args.create_by_index:
        if args.project is None:
            parser.error('--project is required when --create_by_index is given')
        samples = []
        try:
            for ind in args.create_by_index:
                samples.extend(clusters[ind])
        except KeyError:
            parser.error(f'invalid cluster index; Select between from 0 to {len(clusters)}')
        save_project(args.project, args.db, samples)
        logger.info('project %s created with %d cascades', args.project, len(samples))


if __name__ == '__main__':
    with Timer('cluster_cascades'):
        main()
