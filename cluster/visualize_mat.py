import os

import matplotlib.pyplot as plt
from matplotlib import cycler
from scipy import sparse
from scipy.spatial import distance

from sklearn.decomposition import TruncatedSVD

import numpy as np
from settings import BASE_PATH, logger

logger.info('loading the meme-user matrix ...')
loader = np.load(os.path.join(BASE_PATH, 'data/weibo_meme_user_mat.npz'))
mat = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'],
                        dtype=np.float32)
labels = np.load(os.path.join(BASE_PATH, 'data/weibo_meme_labels2.npy'))

logger.info('running truncated SVD ...')
clf = TruncatedSVD(2)
transformed = clf.fit_transform(mat)

labels2 = labels == 5
center = np.array([(6, 0)])
dist = distance.cdist(center, transformed, 'euclidean')
dist = dist.flatten()
labels2 = np.logical_and(labels2, dist < 2)
np.save(os.path.join(BASE_PATH, 'data/weibo_meme_selected.npy'), labels2)


label_values = np.unique(labels2)
ax = plt.subplot(111)
ax.set_prop_cycle(cycler('color',
                         ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                          '#bcbd22', '#17becf']))
for val in label_values:
    indexes = labels2 == val
    ax.scatter(transformed[indexes, 0], transformed[indexes, 1], label='{}: {}'.format(val, np.count_nonzero(indexes)))

plt.legend()
plt.show()
