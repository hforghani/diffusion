import logging
import numpy as np
from scipy import sparse

import settings
from cascade.models import AsLT, ParamTypes
from mongo import mongodb

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('cascade.avg')
logger.setLevel(settings.LOG_LEVEL)


class LTAvg(AsLT):
    def __init__(self, project):
        self.project = project
        self.w_param_name = 'w-avg'
        self.r_param_name = 'r-avg'

        try:
            super(LTAvg, self).__init__(project)
        except FileNotFoundError:
            pass

    def fit(self, calc_weights=True, calc_delays=True, continue_calc=False):
        train_set, test_set = self.project.load_train_test()

        logger.info('querying posts of the memes ...')
        posts_ids = [pm['post_id'] for pm in
                     mongodb.postmemes.find({'meme_id': {'$in': train_set}}, ['post_id']).sort('datetime')]

        if calc_weights or not calc_delays:
            self.calc_weights(posts_ids)
        if calc_delays or not calc_weights:
            self.calc_delays(posts_ids, continue_calc)

    def calc_weights(self, posts_ids):
        logger.info('counting reshares of users ...')
        resh_counts = mongodb.reshares.aggregate([
            {'$match': {'post_id': {'$in': posts_ids}, 'reshared_post_id': {'$in': posts_ids}}},
            {'$group': {'_id': {'user_id': '$user_id', 'ref_user_id': '$ref_user_id'}, 'count': {'$sum': 1}}},
            {'$project': {'user_id': '$_id.user_id', 'ref_user_id': '$_id.ref_user_id', 'count': '$count', '_id': 0}}
        ])

        logger.info('getting users related to the reshares ...')
        users = []
        resh_users = mongodb.reshares.find({'post_id': {'$in': posts_ids}, 'reshared_post_id': {'$in': posts_ids}},
                                           {'user_id': 1, 'ref_user_id': 1, '_id': 0})
        for ru in resh_users:
            users.extend([ru['user_id'], ru['ref_user_id']])
        users = list(set(users))
        # query = {'post_id': {'$in': posts_ids}, 'reshared_post_id': {'$in': posts_ids}}
        # set1 = set(mongodb.reshares.distinct('user_id', query))
        # set2 = set(mongodb.reshares.distinct('ref_user_id', query))
        # users = list(set1 | set2)

        logger.info('counting posts of users ...')
        post_counts = mongodb.posts.aggregate([
            {'$match': {'author_id': {'$in': users}}},
            {'$group': {'_id': '$author_id', 'count': {'$sum': 1}}}])
        post_counts = {obj['_id']: obj['count'] for obj in post_counts}

        logger.info('getting user ids ...')
        user_ids = [u['_id'] for u in mongodb.users.find({}, ['_id']).sort('_id')]
        users_count = len(user_ids)
        users_map = {user_ids[i]: i for i in range(users_count)}

        logger.info('constructing weight matrix ...')
        resh_users.rewind()
        resh_count = resh_users.count()
        values = np.zeros(resh_count)
        ij = np.zeros((2, resh_count))
        i = 0
        for resh in resh_counts:
            values[i] = float(resh['count']) / post_counts[resh['ref_user_id']]
            sender_ind = users_map[resh['ref_user_id']]
            receiver_ind = users_map[resh['user_id']]
            ij[:, i] = [sender_ind, receiver_ind]
            i += 1
            if i % 10000 == 0:
                logger.info('\t%d edges done', i)
        del resh_counts
        weights = sparse.csc_matrix((values, ij), shape=(users_count, users_count))

        logger.info('saving w ...')
        self.project.save_param(weights, self.w_param_name, ParamTypes.SPARSE)
        logger.info('w saved')

    def calc_delays(self, posts_ids, continue_prev):
        logger.info('collecting user ids ...')

        user_ids = [u['_id'] for u in mongodb.users.find({}, ['_id']).sort('_id')]
        user_map = {user_ids[i]: i for i in range(len(user_ids))}
        u_count = len(user_ids)

        r_param_name = 'r-avg'
        index_param_name = 'r-avg-index'
        counts_param_name = 'r-avg-counts'
        r = None
        if continue_prev:
            try:
                logger.info('loading delay temp data ...')
                i = self.project.load_param(index_param_name, ParamTypes.JSON)['index']
                counts = self.project.load_param(counts_param_name, ParamTypes.ARRAY)
                r = self.project.load_param(r_param_name, ParamTypes.ARRAY)
                ignoring = True
                logger.info('ignoring processed reshares ...')
            except:
                logger.info('delay temp data does not exist. extracting from beginning ...')
        if r is None:
            i = 0
            counts = np.zeros(u_count)
            r = np.zeros(u_count, dtype=np.float64)
            ignoring = False

        # Iterate on reshares. Average on the delays of reshares between each pair of users. Save it as diffusion delay
        # between them.
        reshares = mongodb.reshares.find({'post_id': {'$in': posts_ids}, 'reshared_post_id': {'$in': posts_ids}},
                                         {'_id': 0, 'datetime': 1, 'ref_datetime': 1, 'user_id': 1})
        j = 0
        for resh in reshares:
            if ignoring:
                if j < i:
                    j += 1
                    if j % (10 ** 6) == 0:
                        logger.info('\t%d reshares ignored' % j)
                    continue
                else:
                    logger.info('calculating diff. delays ...')
                    ignoring = False
            i += 1
            delay = (resh['datetime'] - resh['ref_datetime']).total_seconds() / (3600.0 * 24)  # in days
            if delay > 0:
                ind = user_map[resh['user_id']]
                if not counts[ind] == 0:
                    r[ind] = delay
                else:
                    r[ind] = (r[ind] * counts[ind] + delay) / (counts[ind] + 1)
                counts[ind] += 1
            if i % 100000 == 0:
                logger.info('\t%d reshares done' % i)

            # Save to continue in the future.
            if i % (10 ** 6) == 0:
                logger.info('saving data ...')
                self.project.save_param({'index': i}, index_param_name, ParamTypes.JSON)
                self.project.save_param(counts, counts_param_name, ParamTypes.ARRAY)
                self.project.save_param(r, r_param_name, ParamTypes.ARRAY)

        logger.info('saving r ...')
        self.project.save_param(r, r_param_name, ParamTypes.ARRAY)
        logger.info('r saved')
