import sys

sys.path.append('.')

import os

import numpy as np
from scipy import sparse
from sklearn.cluster import DBSCAN, KMeans

from settings import logger, BASE_PATH


logger.info('loading the cascade-user matrix ...')
loader = np.load(os.path.join(BASE_PATH, 'data/weibo_cascade_user_mat.npz'))
mat = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'],
                        dtype=np.float32)

# model = DBSCAN(eps=9 * 10 ** 9, min_samples=50, metric='manhattan', metric_params=None, algorithm='auto', leaf_size=30, p=None,
#                n_jobs=-1)
model = KMeans(n_clusters=20, n_jobs=4)
logger.info('clustering ...')
labels = model.fit_predict(mat)

logger.info('clustering done. saving results ...')
np.save(os.path.join(BASE_PATH, 'data/weibo_cascade_labels2.npy'), labels)
