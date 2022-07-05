import math
from functools import reduce

import numpy as np
from scipy import sparse

from diffusion.enum import Method
from diffusion.models import IC
from settings import logger
from utils.time_utils import time_measure


class EMIC(IC):
    method = Method.EMIC
    max_iterations = 20

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold)
        # Do not override k_param_name for EMIC since the precision of parameters will be low when saved.
        self.stop_criterion = 1e-6

    def calc_parameters(self, train_set, project, multi_processed, eco, iterations=None, **kwargs):
        if iterations is None:
            iterations = self.max_iterations

        trees = project.load_trees()
        graph, sequences = project.load_or_extract_graph_seq(train_set)

        user_ids = sorted(graph.nodes())
        u_count = len(user_ids)
        user_map = {user_ids[i]: i for i in range(u_count)}
        cascade_map = {train_set[i]: i for i in range(len(train_set))}
        logger.debug('train set size = %d', len(train_set))
        logger.debug('user space size = %d', len(user_map))

        logger.info('extracting positive indexes and negative counts ...')
        user_times = self._extract_user_times(train_set, user_ids, cascade_map, user_map, sequences, trees)
        edge_indexes = [(user_map[u], user_map[v]) for (u, v) in graph.edges()]
        pos_indexes = {(ui, vi): self._get_pos_indexes(ui, vi, user_times) for (ui, vi) in edge_indexes}
        neg_counts = {(ui, vi): self._get_neg_count(ui, vi, user_times) for (ui, vi) in edge_indexes}

        k = self.__initialize(graph, user_ids, user_map)

        for it in range(iterations):
            logger.info('#%d', it + 1)

            p = self._calc_p(k, train_set, user_ids, user_map, cascade_map, graph, sequences, trees)

            last_k = k.copy()
            k = self._calc_k(p, k, user_ids, user_map, graph, pos_indexes, neg_counts)

            k_dif = np.sqrt((k - last_k).power(2).sum())
            del last_k
            logger.info('k dif = %f', k_dif)
            if k_dif < self.stop_criterion:
                break

        self.k = k.tocsr()

    def _extract_user_times(self, cascade_ids, user_ids, cascade_map, user_map, sequences, trees):
        c_count = len(cascade_ids)
        u_count = len(user_ids)
        values = []
        rows = []
        cols = []

        for cid in cascade_ids:
            c_index = cascade_map[cid]
            tree = trees[cid]
            cur_nodes = tree.roots.copy()
            depth = 0

            while cur_nodes:
                u_indexes = [user_map[node.user_id] for node in cur_nodes if node.user_id in user_map]
                values.extend([depth] * len(u_indexes))
                rows.extend([c_index] * len(u_indexes))
                cols.extend(u_indexes)
                depth += 1
                # Set the next nodes as the children of the current nodes
                cur_nodes = reduce(lambda x, y: x + y, [node.children for node in cur_nodes], [])

        user_times = sparse.csc_matrix((values, (rows, cols)), shape=(c_count, u_count))
        return user_times

    def _get_neg_count(self, sender_index, recv_index, user_times):
        s_times = user_times[:, sender_index]
        r_times = user_times[:, recv_index]
        s_indices = s_times.indices
        r_indices = r_times.indices
        common_indices = list(set(s_indices) & set(r_indices))
        neg_count = s_times.nnz - np.count_nonzero(r_times[common_indices].data == s_times[common_indices].data + 1)
        # logger.debug('sender times: %s', list(zip(s_indices, s_times.data)))
        # logger.debug('receiver times: %s', list(zip(r_indices, r_times.data)))
        # logger.debug('neg_count = %d - %d = %d', s_times.nnz,
        #              np.count_nonzero(r_times[common_indices].data == s_times[common_indices].data + 1), neg_count)
        return neg_count

    def _get_pos_indexes(self, sender_index, recv_index, user_times):
        s_times = user_times[:, sender_index]
        r_times = user_times[:, recv_index]
        s_indices = s_times.indices
        r_indices = r_times.indices
        common_indices = np.array(sorted(set(s_indices) & set(r_indices)))
        pos_indexes = common_indices[r_times[common_indices].data == s_times[common_indices].data + 1]
        # logger.debug('sender times: %s', list(zip(s_indices, s_times.data)))
        # logger.debug('receiver times: %s', list(zip(r_indices, r_times.data)))
        # logger.debug('pos_indexes = %s', pos_indexes)
        return pos_indexes

    def __initialize(self, graph, user_ids, user_map):
        # Initialize probabilities.
        u_count = len(user_ids)
        k = sparse.lil_matrix((u_count, u_count))
        for i in range(u_count):
            child_indexes = [user_map[uid] for uid in graph.successors(user_ids[i])]
            if child_indexes:
                k[i, child_indexes] = 1 / len(child_indexes)
        return k

    @time_measure('debug')
    def _calc_k(self, p, k, user_ids, user_map, graph, pos_indexes, neg_counts):
        u_count = len(user_ids)
        new_k = sparse.lil_matrix((u_count, u_count))
        i = 0
        logger.debug('calculating k: iterating on %d edges', graph.number_of_edges())

        for (u, v) in graph.edges():
            logger.debugv('calculating k: from user %s to %s ...', u, v)
            u_index = user_map[u]
            v_index = user_map[v]
            pos_indexes_u_v = pos_indexes[(u_index, v_index)]
            if pos_indexes_u_v.size == 0:
                new_k[u_index, v_index] = 0
            else:
                neg_count = neg_counts[(u_index, v_index)]
                if pos_indexes_u_v.size == 1:
                    new_k[u_index, v_index] = k[u_index, v_index] / (p[pos_indexes_u_v[0], v_index] * (1 + neg_count))
                else:
                    new_k[u_index, v_index] = k[u_index, v_index] * np.sum(
                        1 / p[pos_indexes_u_v, v_index].toarray()) / (
                                                      pos_indexes_u_v.size + neg_count)
                logger.debugv('negatives count = %d', neg_count)

            logger.debugv('positives count = %d', pos_indexes_u_v.size)
            logger.debugv('pos_indexes = %s', pos_indexes_u_v)
            # logger.debugv('p[pos_indexes, user_index] = %s', p[pos_indexes, v_index])
            logger.debugv('k_u_v = %f', k[u_index, v_index])
            # logger.debugv('last k = %f, new k = %f', k[u_index, v_index], new_k[u_index, v_index])

            i += 1
            if i % 1000 == 0:
                logger.debug('calculating k: %d edges done', i)

        return new_k

    @time_measure('debug')
    def _calc_p(self, k, cascade_ids, user_ids, user_map, cascade_map, graph, sequences, trees):
        c_count = len(cascade_ids)
        u_count = len(user_ids)
        p = sparse.lil_matrix((c_count, u_count))
        i = 0

        for cid in cascade_ids:
            logger.debugv('calculating p: cascade %s ...', cid)
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
                        if len(prev_par_indexes) == 1:
                            p[cindex, vindex] = k[prev_par_indexes[0], vindex]
                        else:
                            p[cindex, vindex] = 1 - np.prod(1 - k[prev_par_indexes, vindex].toarray())
                        logger.debugv('prev_par_indexes = %s', prev_par_indexes)
                        # logger.debugv('k[prev_par_indexes, vindex] = %s', k[prev_par_indexes, vindex])
                        # logger.debugv('p[cindex, vindex] = %f', p[cindex, vindex])
            i += 1
            if i % 100 == 0:
                logger.debug('calculating p: %d cascades done', i)

        return p


class DAIC(EMIC):
    method = Method.DAIC

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, lambdaa=10, **kwargs):
        super().__init__(initial_depth, max_step, threshold)
        # Do not override k_param_name for DAIC since the precision of parameters will be low when saved.
        self.lambdaa = lambdaa

    def _extract_user_times(self, cascade_ids, user_ids, cascade_map, user_map, sequences, trees):
        c_count = len(cascade_ids)
        u_count = len(user_ids)
        users_set = set(user_ids)
        values = []
        rows = []
        cols = []

        for cid in cascade_ids:
            c_index = cascade_map[cid]
            seq = sequences[cid]

            for i in range(len(seq.users)):
                uid = seq.users[i]
                if uid in users_set:
                    u_index = user_map[uid]
                    values.append(seq.times[i])
                    rows.append(c_index)
                    cols.append(u_index)

        user_times = sparse.csc_matrix((values, (rows, cols)), shape=(c_count, u_count))
        return user_times

    @time_measure('debug')
    def _calc_k(self, p, k, user_ids, user_map, graph, pos_indexes, neg_counts):
        u_count = len(user_ids)
        new_k = sparse.lil_matrix((u_count, u_count))
        i = 0
        logger.debug('calculating k: iterating on %d edges', graph.number_of_edges())

        for (u, v) in graph.edges():
            logger.debugv('calculating k: from user %s to %s ...', u, v)
            u_index = user_map[u]
            v_index = user_map[v]
            pos_indexes_u_v = pos_indexes[(u_index, v_index)]
            neg_count = neg_counts[(u_index, v_index)]
            beta = pos_indexes_u_v.size + neg_count + self.lambdaa
            # if (len(pos_indexes_u_v) == 1 and p[pos_indexes_u_v[0], v_index] == 0) \
            #         or (len(pos_indexes_u_v) > 1 and np.any(p[pos_indexes_u_v, v_index].toarray() == 0)):
            #     logger.debug('u_index = %s', u_index)
            #     logger.debug('v_index = %s', v_index)
            #     logger.debug('pos_indexes_u_v = %s', pos_indexes_u_v)
            #     if len(pos_indexes_u_v) == 1:
            #         logger.debug('p of pos indexes = %s', p[pos_indexes_u_v[0], v_index])
            #     else:
            #         logger.debug('p of pos indexes = %s', p[pos_indexes_u_v, v_index].toarray())

            if len(pos_indexes_u_v) == 1:
                gamma = k[u_index, v_index] * 1 / p[pos_indexes_u_v[0], v_index]
            else:
                gamma = k[u_index, v_index] * np.sum(1 / p[pos_indexes_u_v, v_index].toarray())
            delta = beta ** 2 - 4 * self.lambdaa * gamma
            if abs(delta) < 10 ** -10:
                delta = 0
            val = (beta - math.sqrt(delta)) / (2 * self.lambdaa)
            new_k[u_index, v_index] = val

            # logger.debugv('v = %s', v)
            # logger.debugv('pos_indexes_u_v = %s', pos_indexes_u_v)
            # logger.debugv('neg_count = %s', neg_count)
            # logger.debugv('beta = %s', beta)
            # logger.debugv('k[u_index, v_index] = %s', k[u_index, v_index])
            # logger.debugv('p[pos_indexes_u_v, v_index] = %s', p[pos_indexes_u_v, v_index])
            # logger.debugv('np.sum(1 / p[pos_indexes_u_v, v_index] = %s', np.sum(1 / p[pos_indexes_u_v, v_index]))
            # logger.debugv('gamma = %s', gamma)
            # logger.debugv('delta = %s', delta)

            i += 1
            if i % 1000 == 0:
                logger.debug('calculating k: %d edges done', i)

        return new_k

    def _get_neg_count(self, sender_index, recv_index, user_times):
        neg_count = len(set(user_times[:, sender_index].indices) - set(user_times[:, recv_index].indices))
        # logger.debug('sender indices: %s', user_times[:, sender_index].indices)
        # logger.debug('receiver indices: %s', user_times[:, recv_index].indices)
        # logger.debug('neg_count = %s', neg_count)
        return neg_count

    def _get_pos_indexes(self, sender_index, recv_index, user_times):
        s_times = user_times[:, sender_index]
        r_times = user_times[:, recv_index]
        s_indices = s_times.indices
        r_indices = r_times.indices
        common_indices = np.array(sorted(set(s_indices) & set(r_indices)))
        pos_indexes = common_indices[s_times[common_indices].data <= r_times[common_indices].data]
        # logger.debug('sender times: %s', list(zip(s_indices, s_times.data)))
        # logger.debug('receiver times: %s', list(zip(r_indices, r_times.data)))
        # logger.debug('common_indices = %s', common_indices)
        # logger.debug('pos_indexes = %s', pos_indexes)
        return pos_indexes

    @time_measure('debug')
    def _calc_p(self, k, cascade_ids, user_ids, user_map, cascade_map, graph, sequences, trees):
        c_count = len(cascade_ids)
        u_count = len(user_ids)
        p = sparse.lil_matrix((c_count, u_count))
        i = 0

        for cid in cascade_ids:
            logger.debugv('calculating p: cascade %s ...', cid)
            cindex = cascade_map[cid]
            seq = sequences[cid]

            for v in seq.users:
                logger.debugv('calculating p: user %s ...', v)
                if v not in user_map:
                    continue
                vindex = user_map[v]
                prev_par_indexes = [user_map[p] for p in seq.get_active_parents(v, graph)]
                logger.debugv('prev_par_indexes = %s', prev_par_indexes)
                if prev_par_indexes:
                    if len(prev_par_indexes) == 1:
                        logger.debugv('k[prev_par_indexes, vindex] = %s', k[prev_par_indexes[0], vindex])
                        val = k[prev_par_indexes[0], vindex]
                    else:
                        logger.debugv('k[prev_par_indexes, vindex] = %s', k[prev_par_indexes, vindex])
                        val = 1 - np.prod(1 - k[prev_par_indexes, vindex].toarray())
                    logger.debugv('val = %f', val)
                    p[cindex, vindex] = val

            i += 1
            if i % 100 == 0:
                logger.debug('calculating p: %d cascades done', i)

        return p
