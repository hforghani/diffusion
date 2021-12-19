import sys

sys.path.append('.')

import argparse
import json
import os
from bson import ObjectId
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
import numpy as np
from matplotlib import pyplot as plt
import random

from db.managers import DBManager
from settings import logger, BASEPATH, DB_NAME
from cascade.models import Project


def get_users(cascade_id):
    db = DBManager().db
    users = db.postcascades.find({'cascade_id': ObjectId(cascade_id)}, {'author_id': 1, '_id': 0})
    return list({str(u['author_id']) for u in users})


def print_mat_neat(mat):
    count = mat.shape[0]
    print(' ' * 10 + '|', end='')
    for i in range(1, count + 1):
        print('{:<10d}|'.format(i), end='')
    print()

    for i in range(count - 1):
        print('{:<10d}|'.format(i + 1), end='')

        for j in range(i + 1):
            print(' ' * 10 + '|', end='')

        for j in range(i + 1, count):
            print('{:<10f}|'.format(mat[i, j]), end='')
        print()


def print_mat_tab(mat):
    count = mat.shape[0]
    for i in range(1, count + 1):
        print('\t{}'.format(i), end='')
    print()

    for i in range(count - 1):
        print(i + 1, end='')

        for j in range(i + 1):
            print('\t', end='')

        for j in range(i + 1, count):
            print('\t{}'.format(mat[i, j]), end='')
        print()


def print_mat_all_tab(mat):
    count = mat.shape[0]
    for i in range(1, count + 1):
        print('\t{}'.format(i), end='')
    print()

    for i in range(count):
        print(i + 1, end='')
        for j in range(count):
            print('\t{}'.format(mat[i, j]), end='')
        print()


def get_jaccard_mat(cascades, users):
    count = len(cascades)
    mat = np.zeros((count, count))

    for i in range(count - 1):
        users_i = set(users[cascades[i]])

        for j in range(i + 1, count):
            users_j = set(users[cascades[j]])
            # common = len(users_i.intersection(users_j))
            jaccard = len(users_i.intersection(users_j)) / len(users_i.union(users_j))
            mat[i, j] = jaccard

    mat += mat.transpose() + np.eye(count)
    return mat


def heat_map(mat, min_size, max_size, clust_num, method='spectral'):
    fig, ax = plt.subplots()
    # im = ax.imshow(mat)
    # plt.show()
    file_name = os.path.join(BASEPATH, 'data', 'clusters',
                             f'{DB_NAME}-{min_size}to{max_size}-{method}-{clust_num}.png')
    plt.imsave(file_name, mat)


def cluster_mat(mat, clust_num):
    clustering = SpectralClustering(n_clusters=clust_num,
                                    assign_labels="discretize",
                                    random_state=0).fit(mat)
    # clustering = DBSCAN(eps=0.001, min_samples=5, metric='precomputed', n_jobs=-1).fit(mat)
    return clustering.labels_


def load_or_extract_users(cas_ids):
    fname = os.path.join(BASEPATH, 'data', f'{DB_NAME}_users.json')
    try:
        with open(fname) as f:
            logger.info('loading users lists ...')
            users = json.load(f)
    except (FileNotFoundError, ValueError):
        users = {}

    res_users = {}
    i = 1
    changed = False
    for cas_id in cas_ids:
        if cas_id not in users:
            logger.info('querying users of cascade {}'.format(i))
            users[cas_id] = get_users(cas_id)
            changed = True
        res_users[cas_id] = users[cas_id]
        i += 1

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
        clust_size = clusters[label].size
        squares_mat[begin: begin + clust_size, begin: begin + clust_size] = 1
        begin += clust_size
    error = np.linalg.norm(mat - squares_mat)
    return error


def show_clusters(clust_num, min_size, max_size, depth):
    # Extract the top cascades.
    db = DBManager().db
    query = {}
    if min_size is not None:
        query['size'] = {'$gte': min_size}
    if max_size is not None:
        query.setdefault('size', {})
        query['size']['$lte'] = max_size
    if depth is not None:
        query['depth'] = {'$gte': depth}
    res_cascades = list(db.cascades.find(query, ['_id', 'size']))
    cascades = [str(m['_id']) for m in res_cascades]
    sizes = {str(m['_id']): m['size'] for m in res_cascades}
    logger.info('%d cascades found', len(cascades))

    # Extract user sets of top cascades
    users = load_or_extract_users(cascades)

    # Calculate the Jaccard matrix.
    logger.info('creating the similarity matrix ...')
    mat = get_jaccard_mat(cascades, users)

    # Normalize the matrix.
    mat = np.reshape(mat - np.eye(len(cascades)), (1, mat.size))
    mat = normalize(mat, norm='max')
    mat = np.reshape(mat, (len(cascades), len(cascades)))
    mat += np.eye(len(cascades))

    # Cluster the cascades.
    logger.info('clustering the cascades ...')
    labels = cluster_mat(mat, clust_num)

    # Create the ordered indexes of cascades.
    ordered_ind = np.array([], dtype=np.int64)
    cascades_arr = np.array([str(m) for m in cascades])
    uni_val = np.unique(labels)
    clusters = {}
    logger.info('%d cluster(s) found', uni_val.size)
    for val in uni_val:
        indexes = np.nonzero(labels == val)[0]
        indexes = indexes.astype(np.int64)
        clusters[val] = cascades_arr[indexes]
        ordered_ind = np.concatenate((ordered_ind, indexes))

    # Print the clusters into the file.
    method = 'spectral'
    file_name = os.path.join(BASEPATH, 'data', 'clusters',
                             f'{DB_NAME}-{min_size}to{max_size}-{method}-{clust_num}')
    with open(os.path.join(BASEPATH, 'data', file_name), 'w') as f:
        for label in sorted(clusters.keys()):
            clust_cascades = clusters[label]
            f.write(f'cluster {label}: count = {clust_cascades.size}\n')
            f.write('\n'.join([f'{cas_id}, size = {sizes[cas_id]}' for cas_id in clust_cascades]))
            f.write('\n')

    new_mat = mat[:, ordered_ind]
    new_mat = new_mat[ordered_ind, :]

    # Calculate the clustering error.
    error = calc_error(new_mat, clusters)
    logger.info('error = %f', error)

    # print_mat_all_tab(new_mat)
    heat_map(new_mat, min_size, max_size, clust_num, method)

    return clusters


def ask_question(question, choices):
    while True:
        ans = input(f'{question} ({choices}) ')
        lower_choices = [c.lower() for c in choices]
        if ans.lower() in lower_choices:
            return ans.lower()
        else:
            print('Wrong answer!')


def save_project(name, cas_ids):
    random.shuffle(cas_ids)
    val_ratio, test_ratio = 0.15, 0.15
    val_num = round(val_ratio * len(cas_ids))
    test_num = round(test_ratio * len(cas_ids))
    val_set = cas_ids[:val_num]
    test_set = cas_ids[val_num:val_num + test_num]
    train_set = cas_ids[val_num + test_num:]
    project = Project(name)
    project.save_sets(train_set, val_set, test_set)


def ask_to_create_project(clusters):
    ans = ask_question('Do you want to peek a cluster as a project?', {'y': 'Yes', 'n': 'No'})
    if ans == 'n':
        return
    label = ask_question('Which cluster do you want to peek?',
                         {str(k): f'of size {len(clusters[k])}' for k in sorted(clusters.keys())})
    name = input('Enter the project name: ')
    save_project(name, list(clusters[int(label)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cluster the cascades based on their common users')
    parser.add_argument('-c', '--clusters', type=int, default=4, help='number of clusters')
    parser.add_argument('-m', '--min', type=int, help='minimum cascade size')
    parser.add_argument('-x', '--max', type=int, help='minimum cascade size')
    parser.add_argument('-d', '--depth', type=int, help='minimum depth')
    args = parser.parse_args()
    clusters = show_clusters(args.clusters, args.min, args.max, args.depth)
    ask_to_create_project(clusters)
