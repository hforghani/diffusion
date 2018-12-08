import logging
from django.db.models import Count
import numpy as np
from scipy import sparse

from cascade.models import AsLT, ParamTypes
from crud.models import UserAccount, Post, Reshare

logger = logging.getLogger('cascade.avg')


class LTAvg(AsLT):
    def __init__(self, project):
        self.project = project
        self.w_param_name = 'w-avg'
        self.r_param_name = 'r-avg'
        super(LTAvg, self).__init__(project)

    def fit(self, calc_weights=True, calc_delays=True, continue_calc=False):
        train_set, test_set = self.project.load_train_test()

        logger.info('querying posts and reshares ...')
        posts = Post.objects.filter(postmeme__meme_id__in=train_set).distinct().order_by('datetime')
        reshares = Reshare.objects.filter(post__in=posts, reshared_post__in=posts).distinct()

        if calc_weights or not calc_delays:
            self.calc_weights(reshares, posts)
        if calc_delays or not calc_weights:
            self.calc_delays(reshares, continue_calc)

    def calc_weights(self, reshares, posts):
        logger.info('counting reshares of users ...')
        resh_counts = reshares.values('user_id', 'ref_user_id').annotate(count=Count('datetime'))
        # resh_counts = reshares.values('post__author_id', 'reshared_post__author_id').annotate(count=Count('datetime'))

        logger.info('counting posts of users ...')
        post_counts = list(posts.values('author_id').annotate(count=Count('datetime')))
        post_counts = {obj['author_id']: obj['count'] for obj in post_counts}

        logger.info('getting user ids ...')
        user_ids = UserAccount.objects.values_list('id', flat=True)
        users_count = len(user_ids)
        users_map = {user_ids[i]: i for i in range(users_count)}

        logger.info('constructing weight matrix ...')
        resh_count = resh_counts.count()
        values = np.zeros(resh_count)
        ij = np.zeros((2, resh_count))
        i = 0
        for resh in resh_counts:
            values[i] = float(resh['count']) / post_counts[resh['ref_user_id']]
            # values[i] = float(resh['count']) / post_counts[resh['reshared_post__author_id']]
            sender_id = users_map[resh['ref_user_id']]
            # sender_id = users_map[resh['reshared_post__author_id']]
            receiver_id = users_map[resh['user_id']]
            # receiver_id = users_map[resh['post__author_id']]
            ij[:, i] = [sender_id, receiver_id]
            i += 1
            if i % 10000 == 0:
                logger.info('\t%d edges done' % i)
        del resh_counts
        weights = sparse.csc_matrix((values, ij), shape=(users_count, users_count))

        logger.info('saving w ...')
        self.project.save_param(weights, self.w_param_name, ParamTypes.SPARSE)
        logger.info('w saved')

    def calc_delays(self, reshares, continue_prev):
        logger.info('collecting user ids ...')

        user_ids = UserAccount.objects.values_list('id', flat=True).order_by('id')
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
        j = 0
        for resh in reshares.iterator():
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
            delay = (resh.datetime - resh.ref_datetime).total_seconds() / (3600.0 * 24)  # in days
            # delay = (resh.datetime - resh.reshared_post.datetime).total_seconds() / (3600.0 * 24)  # in days
            if delay > 0:
                ind = user_map[resh.user_id]
                # ind = user_map[resh.post.author_id]
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
