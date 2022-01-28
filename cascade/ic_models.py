import math
import pprint
from functools import reduce

import numpy as np
from scipy import sparse

import settings
from cascade.models import IC, ParamTypes
from log_levels import DEBUG_LEVELV_NUM
from settings import logger
from utils.time_utils import time_measure


class EMIC(IC):
    def __init__(self, project):
        super(EMIC, self).__init__(project)
        self.k_param_name = 'k-emic'
        self.max_iterations = 15

    def calc_parameters(self, train_set, multi_processed, eco, **kwargs):
        iterations = kwargs.get('iterations', self.max_iterations)
        if iterations is None:
            iterations = self.max_iterations

        graph, sequences = self.project.load_or_extract_graph_seq()
        trees = self.project.load_trees()

        user_ids = sorted(graph.nodes())
        u_count = len(user_ids)
        user_map = {user_ids[i]: i for i in range(u_count)}
        cascade_map = {train_set[i]: i for i in range(len(train_set))}
        logger.info('train set size = %d', len(train_set))
        logger.info('user space size = %d', len(user_map))

        user_times = self._extract_user_times(train_set, user_ids, cascade_map, user_map, sequences, trees)

        k = self.__initialize(graph, user_ids, user_map)

        for it in range(iterations):
            logger.info('#%d', it + 1)

            p = self.__calc_p(k, train_set, user_ids, user_map, cascade_map, graph, sequences, trees)

            last_k = k.copy()
            k = self._calc_k(p, k, user_ids, user_map, graph, user_times)

            k_dif = k - last_k
            k_dif = np.sqrt(np.multiply(k_dif, k_dif).sum())
            del last_k
            logger.info('k dif = %f', k_dif)

            if eco:
                self.project.save_param(sparse.csr_matrix(k), self.k_param_name, ParamTypes.SPARSE)
                logger.debug('k parameters saved')

        self.k = k

    def _extract_user_times(self, cascade_ids, user_ids, cascade_map, user_map, sequences, trees):
        c_count = len(cascade_ids)
        u_count = len(user_ids)
        user_times = np.zeros((c_count, u_count))
        user_times[:] = np.nan

        for cid in cascade_ids:
            c_index = cascade_map[cid]
            tree = trees[cid]
            cur_nodes = tree.roots.copy()
            depth = 0

            while cur_nodes:
                u_indexes = [user_map[node.user_id] for node in cur_nodes if node.user_id in user_map]
                user_times[c_index, u_indexes] = depth
                depth += 1
                # Set the next nodes as the children of the current nodes
                cur_nodes = reduce(lambda x, y: x + y, [node.children for node in cur_nodes], [])

        return user_times

    def __initialize(self, graph, user_ids, user_map):
        # Initialize probabilities.
        u_count = len(user_ids)
        k = np.zeros((u_count, u_count))
        for i in range(u_count):
            child_indexes = [user_map[uid] for uid in graph.successors(user_ids[i])]
            if child_indexes:
                k[i, child_indexes] = 1 / len(child_indexes)
        return k

    @time_measure('debug')
    def _calc_k(self, p, k, user_ids, user_map, graph, user_times):
        u_count = len(user_ids)
        new_k = np.zeros((u_count, u_count))

        for u in user_ids:
            logger.debugv('calculating k: from user %s ...', u)
            u_index = user_map[u]

            for v in graph.successors(u):
                logger.debugv('calculating k: to user %s ...', v)
                v_index = user_map[v]
                pos_indexes = np.nonzero(~np.isnan(user_times[:, u_index]) & ~np.isnan(user_times[:, v_index]) & (
                        user_times[:, u_index] == user_times[:, v_index] - 1))[0]
                if pos_indexes.size == 0:
                    new_k[u_index, v_index] = 0
                else:
                    neg_count = np.count_nonzero(~np.isnan(user_times[:, u_index]) & (
                            np.isnan(user_times[:, v_index]) | (user_times[:, u_index] != user_times[:, v_index] - 1)))
                    new_k[u_index, v_index] = k[u_index, v_index] * np.sum(1 / p[pos_indexes, v_index]) / (
                            len(pos_indexes) + neg_count)
                    logger.debugv('negatives count = %d', neg_count)

                logger.debugv('positives count = %d', len(pos_indexes))
                logger.debugv('pos_indexes = %s', pos_indexes)
                # logger.debugv('p[pos_indexes, user_index] = %s', p[pos_indexes, v_index])
                logger.debugv('k_u_v = %f', k[u_index, v_index])
                # logger.debugv('last k = %f, new k = %f', k[u_index, v_index], new_k[u_index, v_index])

        return new_k

    @time_measure('debug')
    def __calc_p(self, k, cascade_ids, user_ids, user_map, cascade_map, graph, sequences, trees):
        c_count = len(cascade_ids)
        u_count = len(user_ids)
        p = np.zeros((c_count, u_count))

        for cid in cascade_ids:
            logger.debug('calculating p: cascade %s ...', cid)
            cindex = cascade_map[cid]
            tree = trees[cid]

            for v in tree.node_ids():
                logger.debugv('calculating p: user %s ...', v)
                if v not in user_map:
                    continue
                vindex = user_map[v]
                depth = tree.depth_of(v)
                logger.debugv('depth = %d', depth)
                if depth > 0:
                    parents = set(graph.predecessors(v))
                    prev_step = set([node.user_id for node in tree.nodes_at_depth(depth - 1)])
                    # logger.debugv('parents =\n%s', pprint.pformat(parents))
                    # logger.debugv('previous step users =\n%s', pprint.pformat(prev_step))
                    prev_par_indexes = [user_map[u] for u in parents & prev_step]
                    if prev_par_indexes:
                        p[cindex, vindex] = 1 - np.prod(1 - k[prev_par_indexes, vindex])
                        logger.debugv('prev_par_indexes = %s', prev_par_indexes)
                        # logger.debugv('k[prev_par_indexes, vindex] = %s', k[prev_par_indexes, vindex])
                        # logger.debugv('p[cindex, vindex] = %f', p[cindex, vindex])

        return p


class DAIC(EMIC):
    def __init__(self, project):
        super(DAIC, self).__init__(project)
        self.k_param_name = 'k-daic'
        self.max_iterations = 15

    def _extract_user_times(self, cascade_ids, user_ids, cascade_map, user_map, sequences, trees):
        c_count = len(cascade_ids)
        u_count = len(user_ids)
        users_set = set(user_ids)
        user_times = np.zeros((c_count, u_count))
        user_times[:] = np.nan

        for cid in cascade_ids:
            c_index = cascade_map[cid]
            seq = sequences[cid]

            for i in range(len(seq.users)):
                uid = seq.users[i]
                if uid in users_set:
                    u_index = user_map[uid]
                    user_times[c_index, u_index] = seq.times[i]

        return user_times

    @time_measure('debug')
    def _calc_k(self, p, k, user_ids, user_map, graph, user_times):
        u_count = len(user_ids)
        new_k = np.zeros((u_count, u_count))
        lambdaa = 10

        for u in user_ids:
            logger.debugv('calculating k: from user %s ...', u)
            u_index = user_map[u]

            for v in graph.successors(u):
                logger.debugv('calculating k: to user %s ...', v)
                v_index = user_map[v]
                pos_indexes = np.nonzero(~np.isnan(user_times[:, u_index]) & ~np.isnan(user_times[:, v_index]) & (
                        user_times[:, u_index] < user_times[:, v_index]))[0]
                neg_count = np.count_nonzero(~np.isnan(user_times[:, u_index]) & np.isnan(user_times[:, v_index]))
                beta = pos_indexes.size + neg_count + lambdaa
                gamma = k[u_index, v_index] * np.sum(1 / p[pos_indexes, v_index])
                delta = beta ** 2 - 4 * lambdaa * gamma
                new_k[u_index, v_index] = (beta - math.sqrt(delta)) / (2 * lambdaa)

                logger.debugv('positives count = %d', len(pos_indexes))
                logger.debugv('negatives count = %d', neg_count)
                logger.debugv('beta = %s', beta)
                logger.debugv('pos_indexes = %s', pos_indexes)
                # logger.debugv('p[pos_indexes, user_index] = %s', p[pos_indexes, user_index])
                # logger.debugv('k_u_v = %f', k[u_index, v_index])
                logger.debugv('gamma = %f', gamma)
                logger.debugv('delta = %f', delta)
                # logger.debugv('last k = %f, new k = %f', k[u_index, v_index], new_k[u_index, v_index])

        return new_k

    @time_measure('debug')
    def __calc_p(self, theta, cascade_ids, user_ids, user_map, cascade_map, graph, sequences, trees):
        c_count = len(cascade_ids)
        u_count = len(user_ids)
        p = np.zeros((c_count, u_count))

        for cid in cascade_ids:
            logger.debug('calculating p: cascade %s ...', cid)
            cindex = cascade_map[cid]
            seq = sequences[cid]

            for v in seq.users:
                logger.debugv('calculating p: user %s ...', v)
                if v not in user_map:
                    continue
                vindex = user_map[v]
                prev_par_indexes = [user_map[p] for p in seq.get_active_parents(v, graph)]
                if prev_par_indexes:
                    p[cindex, vindex] = 1 - np.prod(1 - theta[prev_par_indexes, vindex])
                    logger.debugv('prev_par_indexes = %s', prev_par_indexes)
                    # logger.debugv('theta[prev_par_indexes, vindex] = %s', theta[prev_par_indexes, vindex])
                    # logger.debugv('p[cindex, vindex] = %f', p[cindex, vindex])

        return p
