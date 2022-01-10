import numpy as np
from scipy import sparse

from db.managers import DBManager
from settings import logger
from cascade.models import LT, ParamTypes


class LTAvg(LT):
    def __init__(self, project):
        self.project = project
        self.w_param_name = 'w-avg'
        self.r_param_name = 'r-avg'

        try:
            super(LTAvg, self).__init__(project)
        except FileNotFoundError:
            pass

    def fit(self, calc_weights=True, calc_delays=True, continue_calc=False):
        try:
            super().fit()
        except FileNotFoundError:
            train_set, _, _ = self.project.load_sets()

            logger.info('querying posts of the cascades ...')
            db = DBManager(self.project.db).db
            posts_ids = [pc['post_id'] for pc in
                         db.postcascades.find({'cascade_id': {'$in': train_set}}, ['post_id']).sort('datetime')]

            logger.info('getting user ids ...')
            user_ids = [u['_id'] for u in db.users.find({}, ['_id']).sort('_id')]
            users_count = len(user_ids)
            users_map = {user_ids[i]: i for i in range(users_count)}

            if calc_weights or not calc_delays:
                self.calc_weights(posts_ids, users_map)
            if calc_delays or not calc_weights:
                self.calc_delays(posts_ids, users_map, continue_calc)

        return self

    def calc_weights(self, posts_ids, users_map):
        logger.info('counting reshares of users ...')
        db = DBManager(self.project.db).db
        resh_counts = db.reshares.aggregate([
            {'$match': {'post_id': {'$in': posts_ids}, 'reshared_post_id': {'$in': posts_ids}}},
            {'$group': {'_id': {'user_id': '$user_id', 'ref_user_id': '$ref_user_id'}, 'count': {'$sum': 1}}},
            {'$project': {'user_id': '$_id.user_id', 'ref_user_id': '$_id.ref_user_id', 'count': '$count', '_id': 0}}
        ])

        logger.info('getting users related to the reshares ...')
        users = []
        resh_users = db.reshares.find({'post_id': {'$in': posts_ids}, 'reshared_post_id': {'$in': posts_ids}},
                                      {'user_id': 1, 'ref_user_id': 1, '_id': 0})
        for ru in resh_users:
            users.extend([ru['user_id'], ru['ref_user_id']])
        users = list(set(users))

        logger.info('counting posts of users ...')
        post_counts = db.posts.aggregate([
            {'$match': {'author_id': {'$in': users}}},
            {'$group': {'_id': '$author_id', 'count': {'$sum': 1}}}])
        post_counts = {obj['_id']: obj['count'] for obj in post_counts}

        logger.info('constructing weight matrix ...')
        resh_users.rewind()
        resh_count = resh_users.count()
        values = np.zeros(resh_count)
        ij = np.zeros((2, resh_count))
        i = 0
        for resh in resh_counts:
            values[i] = resh['count'] / post_counts[resh['ref_user_id']]
            sender_ind = users_map[resh['ref_user_id']]
            receiver_ind = users_map[resh['user_id']]
            ij[:, i] = [sender_ind, receiver_ind]
            logger.debug('%s to %s: reshares count = %d, i posts count = %d, weight = %f', resh['ref_user_id'],
                         resh['user_id'], resh['count'], post_counts[resh['ref_user_id']], values[i])
            i += 1
            if i % 10000 == 0:
                logger.info('\t%d edges done', i)
        del resh_counts
        logger.debug('max weight = %f', np.max(values))
        users_count = len(users_map)
        values /= np.max(values)
        weights = sparse.csc_matrix((values, ij), shape=(users_count, users_count))

        logger.info('saving w ...')
        self.project.save_param(weights, self.w_param_name, ParamTypes.SPARSE)
        self.w = weights.tocsr()
        logger.info('w saved')

    def calc_delays(self, posts_ids, users_map, continue_prev):
        u_count = len(users_map)

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
        db = DBManager(self.project.db).db
        reshares = db.reshares.find({'post_id': {'$in': posts_ids}, 'reshared_post_id': {'$in': posts_ids}},
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
            if resh['ref_datetime'] is not None:
                delay = (resh['datetime'] - resh['ref_datetime']).total_seconds() / (3600.0 * 24)  # in days
                ind = users_map[resh['user_id']]
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
        self.r = r
        logger.info('r saved')
