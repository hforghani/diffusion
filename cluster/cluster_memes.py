import json
import os
from bson import ObjectId
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.preprocessing.data import normalize
from settings import mongodb, logger, BASEPATH
import numpy as np
from matplotlib import pyplot as plt


def get_users(meme_id):
    #logger.info('querying posts ...')
    posts = [pm['post_id'] for pm in mongodb.postmemes.find({'meme_id': ObjectId(meme_id)}, {'post_id': 1, '_id': 0})]
    #logger.info('size of posts: {}'.format(len(posts)))
    #logger.info('querying users ...')
    users = [p['author_id'] for p in mongodb.posts.find({'_id': {'$in': posts}}, {'author_id': 1, '_id': 0})]
    return list({str(u) for u in users})


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


def get_jaccard_mat(memes, users):
    count = len(memes)
    mat = np.zeros((count, count))

    for i in range(count - 1):
        users_i = set(users[memes[i]])

        for j in range(i + 1, count):
            users_j = set(users[memes[j]])
            #common = len(users_i.intersection(users_j))
            jaccard = len(users_i.intersection(users_j)) / len(users_i.union(users_j))
            mat[i, j] = jaccard

    mat += mat.transpose() + np.eye(count)
    return mat


def heat_map(mat):
    fig, ax = plt.subplots()
    im = ax.imshow(mat)
    plt.show()


def cluster_mat(mat, clust_num):
    clustering = SpectralClustering(n_clusters=clust_num,
                                    assign_labels="discretize",
                                    random_state=0).fit(mat)
    #clustering = DBSCAN(eps=0.001, min_samples=5, metric='precomputed', n_jobs=-1).fit(mat)
    return clustering.labels_


def load_or_extract_users(meme_ids):
    fname = os.path.join(BASEPATH, 'data', 'weibo_users.json')
    try:
        with open(fname) as f:
            logger.info('loading users lists ...')
            users = json.load(f)
    except (FileNotFoundError, ValueError):
        users = {}

    res_users = {}
    i = 1
    changed = False
    for meme_id in meme_ids:
        if meme_id not in users:
            logger.info('querying users of cascade {}'.format(i))
            users[meme_id] = get_users(meme_id)
            changed = True
        res_users[meme_id] = users[meme_id]
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
    for clust, clust_memes in clusters:
        clust_size = clust_memes.size
        squares_mat[begin: begin + clust_size, begin: begin + clust_size] = 1
        begin += clust_size
    error = np.linalg.norm(mat - squares_mat)
    return error


def main():
    count = 100
    clust_num = 4

    # Extract the top cascades.
    memes = [str(m['_id']) for m in mongodb.memes.find({}, ['_id']).sort('count', -1)[:count]]

    # Extract user sets of top cascades
    users = load_or_extract_users(memes)

    # Calculate the Jaccard matrix.
    logger.info('creating the similarity matrix ...')
    mat = get_jaccard_mat(memes, users)
    mat = normalize(mat - np.eye(count), norm='max') + np.eye(count)

    # Cluster the cascades.
    logger.info('clustering the cascades ...')
    labels = cluster_mat(mat, clust_num)

    # Create the ordered indexes of cascades.
    ordered_ind = np.array([], dtype=np.int8)
    memes_arr = np.array([str(m) for m in memes])
    uni_val = np.unique(labels)
    clusters = []
    logger.info('%d cluster(s) found', uni_val.size)
    for val in uni_val:
        indexes = np.nonzero(labels == val)[0]
        indexes = indexes.astype(np.int8)
        clusters.append((val, memes_arr[indexes]))
        ordered_ind = np.concatenate((ordered_ind, indexes))

    # Print the clusters into the file.
    with open(os.path.join(BASEPATH, 'data', 'weibo-clust'), 'w') as f:
        for clust, clust_memes in clusters:
            f.write('cluster {}: size = {}\n'.format(clust, clust_memes.size))
            f.write('\n'.join(clust_memes))
            f.write('\n')

    new_mat = mat[:, ordered_ind]
    new_mat = new_mat[ordered_ind, :]

    # Calculate the clustering error.
    error = calc_error(new_mat, clusters)
    logger.info('error = %f', error)

    #print_mat_all_tab(new_mat)
    heat_map(new_mat)


if __name__ == '__main__':
    main()
