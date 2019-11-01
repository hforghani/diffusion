from scipy import sparse
import numpy as np
from settings import mongodb, logger
from utils.numpy_utils import save_sparse


logger.info('mapping meme ids to indexes ...')
meme_ids = [m['_id'] for m in mongodb.memes.find({}, ['_id'], no_cursor_timeout=True).sort('_id')]
meme_map = {str(meme_ids[i]): i for i in range(len(meme_ids))}
m_count = len(meme_ids)
del meme_ids

logger.info('mapping user ids to indexes ...')
user_ids = [u['_id'] for u in mongodb.users.find({}, ['_id'], no_cursor_timeout=True).sort('_id')]
user_map = {str(user_ids[i]): i for i in range(len(user_ids))}
u_count = len(user_ids)
del user_ids

#meme_user_mat = sparse.lil_matrix((m_count, u_count), dtype=bool)

logger.info('mapping posts to authors ...')
post_author_map = {str(u['_id']): user_map[str(u['author_id'])] for u in
                   mongodb.posts.find({}, {'_id', 'author_id'}, no_cursor_timeout=True)}
del user_map

logger.info('reading postmemes ...')
meme_users = set()
i = 0

for pm in mongodb.postmemes.find({}, {'_id': 0, 'post_id': 1, 'meme_id': 1}, no_cursor_timeout=True):
    post_id = str(pm['post_id'])
    user_ind = post_author_map[post_id]
    meme_id = str(pm['meme_id'])
    meme_users.add((meme_map[meme_id], user_ind))
    #meme_user_mat[meme_map[meme_id], user_map[user_id]] = 1

    i += 1
    if i % 1000 == 0:
        logger.info('%d postmemes read', i)

del post_author_map
del meme_map

#logger.info('converting to csr ...')
#meme_user_mat = meme_user_mat.tocsr()

logger.info('creating matrix ...')
row_ind = []
col_ind = []
for r, c in meme_users:
    row_ind.append(r)
    col_ind.append(c)
del meme_users
meme_user_mat = sparse.csr_matrix((np.ones(len(row_ind)), (np.array(row_ind), np.array(col_ind))),
                                  shape=(m_count, u_count), dtype=bool)
del row_ind
del col_ind

logger.info('saving into file ...')
save_sparse('data/weibo_meme_user_mat.npz', meme_user_mat)

#results = mongodb.postmemes.aggregate([
#                                          {'$lookup':
#                                               {
#                                                   'from': 'posts',
#                                                   'localField': 'post_id',
#                                                   'foreignField': '_id',
#                                                   'as': 'post'
#                                               }
#                                          },
#                                          {'$project': {
#                                              'meme_id': 1, 'post_id': 1, 'post.author_id': 1
#                                          }},
#                                          {'$group': {
#                                              '_id': {'author_id': '$post.author_id', 'meme_id': '$meme_id'},
#                                              'count': {'$sum': 1}
#                                          }},
#                                      ], allowDiskUse=True)
#
#logger.info('saving into file ...')
#with open('weibo_meme_author_counts.json', 'w') as f:
#    json.dump(results, f)
