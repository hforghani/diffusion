import sys

sys.path.append('.')

from scipy import sparse
import numpy as np

from db.managers import DBManager
from settings import logger, DB_NAME
from utils.numpy_utils import save_sparse

logger.info('mapping cascade ids to indexes ...')
db = DBManager().db
cursor = db.cascades.find({}, ['_id'], no_cursor_timeout=True).sort('_id')
cascade_ids = [m['_id'] for m in cursor]
cursor.close()
cascade_map = {str(cascade_ids[i]): i for i in range(len(cascade_ids))}
m_count = len(cascade_ids)
del cascade_ids

logger.info('mapping user ids to indexes ...')
cursor = db.users.find({}, ['_id'], no_cursor_timeout=True).sort('_id')
user_ids = [u['_id'] for u in cursor]
cursor.close()
user_map = {str(user_ids[i]): i for i in range(len(user_ids))}
u_count = len(user_ids)
del user_ids

# cascade_user_mat = sparse.lil_matrix((m_count, u_count), dtype=bool)

logger.info('mapping posts to authors ...')
cursor = db.posts.find({}, {'_id', 'author_id'}, no_cursor_timeout=True)
post_author_map = {str(u['_id']): user_map[str(u['author_id'])] for u in cursor}
cursor.close()
del user_map

logger.info('reading postcascades ...')
cascade_users = set()
i = 0

cursor = db.postcascades.find({}, {'_id': 0, 'post_id': 1, 'cascade_id': 1}, no_cursor_timeout=True)
for pm in cursor:
    post_id = str(pm['post_id'])
    user_ind = post_author_map[post_id]
    cascade_id = str(pm['cascade_id'])
    cascade_users.add((cascade_map[cascade_id], user_ind))
    # cascade_user_mat[cascade_map[cascade_id], user_map[user_id]] = 1

    i += 1
    if i % 1000 == 0:
        logger.info('%d postcascades read', i)

cursor.close()
del post_author_map
del cascade_map

# logger.info('converting to csr ...')
# cascade_user_mat = cascade_user_mat.tocsr()

logger.info('creating matrix ...')
row_ind = []
col_ind = []
for r, c in cascade_users:
    row_ind.append(r)
    col_ind.append(c)
del cascade_users
cascade_user_mat = sparse.csr_matrix((np.ones(len(row_ind)), (np.array(row_ind), np.array(col_ind))),
                                     shape=(m_count, u_count), dtype=bool)
del row_ind
del col_ind

logger.info('saving into file ...')
save_sparse(f'../data/{DB_NAME}_cascade_user_mat.npz', cascade_user_mat)
