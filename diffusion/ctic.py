import math
import traceback
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import Pool

import numpy as np
from bson import ObjectId
from scipy import sparse

import settings
from cascade.models import ParamTypes
from diffusion.enum import Method
from diffusion.models import IC
from settings import logger
from utils.time_utils import Timer, time_measure


def calc_a(sequences, graph, k, r, user_map):
    try:
        u_count = len(user_map)
        c_count = len(sequences)
        a = {}
        i = 0

        for cid, seq in sequences.items():
            values = []
            rows = []
            cols = []

            for v in seq.users:
                if v not in user_map:
                    continue
                v_index = user_map[v]
                parents = seq.get_active_parents(v, graph)
                if not parents:
                    continue
                parents_indexes = [user_map[uid] for uid in parents]
                parents_times = np.array([seq.user_times[vid] for vid in parents])
                user_time = np.ones(len(parents)) * seq.user_times[v]
                diff = (user_time - parents_times).reshape(len(parents), 1)
                diff[diff == 0] = 1.0 / (30 * 24 * 60)  # 1 minute
                k_col = k[parents_indexes, v_index].toarray()
                r_col = r[parents_indexes, v_index].toarray()
                val = np.multiply(np.multiply(k_col, r_col), np.exp(-np.multiply(r_col, diff)))
                # if (np.float32(val) == 0).any():
                # logger.debug('cid = %s, v = %s', cid, v)
                # logger.debug('parents = %s', parents)
                # logger.debug('diff = %s', diff)
                # logger.debug('k_col = %s', k_col)
                # logger.debug('r_col = %s', r_col)
                # logger.debug('val = %s', val)
                # logger.warning('\ta = 0')
                if val.size > 1:
                    values.extend(np.squeeze(val).tolist())
                else:
                    values.append(float(val))
                rows.extend(parents_indexes)
                cols.extend([v_index] * len(parents_indexes))

            a[cid] = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float64)

            i += 1
            if c_count >= 10 and i % (c_count // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // c_count)

        return a
    except:
        logger.error(traceback.format_exc())
        raise


def calc_b(sequences, graph, k, r, user_map):
    try:
        u_count = len(user_map)
        c_count = len(sequences)
        b = {}
        i = 0

        for cid, seq in sequences.items():
            values = []
            rows = []
            cols = []

            for v in seq.users:
                if v not in user_map:
                    continue
                v_index = user_map[v]
                parents = seq.get_active_parents(v, graph)
                if not parents:
                    continue
                parents_indexes = [user_map[uid] for uid in parents]
                parents_times = np.array([seq.user_times[vid] for vid in parents])
                user_time = np.ones(len(parents)) * seq.user_times[v]
                diff = (user_time - parents_times).reshape(len(parents), 1)
                diff[diff == 0] = 1.0 / (30 * 24 * 60)  # 1 minute
                k_col = k[parents_indexes, v_index].toarray()
                r_col = r[parents_indexes, v_index].toarray()
                val = np.multiply(k_col, np.exp(-np.multiply(r_col, diff))) + 1 - k_col
                if (np.float32(val) == 0).any():
                    logger.warning('\tb = 0')
                if val.size > 1:
                    values.extend(np.squeeze(val).tolist())
                else:
                    values.append(float(val))
                rows.extend(parents_indexes)
                cols.extend([v_index] * len(parents_indexes))

            b[cid] = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float64)

            i += 1
            if c_count >= 10 and i % (c_count // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // c_count)

        return b
    except:
        logger.error(traceback.format_exc())
        raise


def calc_alpha(sequences, graph, k, r, user_map):
    try:
        u_count = len(user_map)
        c_count = len(sequences)
        alpha = {}
        i = 0

        for cid, seq in sequences.items():
            values = []
            rows = []
            cols = []

            for v in seq.users:
                if v not in user_map:
                    continue
                v_index = user_map[v]
                parents = seq.get_active_parents(v, graph)
                if not parents:
                    continue

                parents_indexes = [user_map[uid] for uid in parents]
                if len(parents) == 1:
                    values.append(1)
                else:
                    parents_times = np.array([seq.user_times[vid] for vid in parents])
                    user_time = np.ones(len(parents)) * seq.user_times[v]
                    diff = (user_time - parents_times).reshape(len(parents), 1)
                    diff[diff == 0] = 1.0 / (30 * 24 * 60)  # 1 minute
                    k_col = k[parents_indexes, v_index].toarray()
                    r_col = r[parents_indexes, v_index].toarray()
                    k_exp = np.multiply(k_col, np.exp(-np.multiply(r_col, diff)))
                    val = np.divide(np.multiply(r_col, k_exp), k_exp + 1 - k_col)
                    val /= np.sum(val)
                    if val.size > 1:
                        values.extend(np.squeeze(val).tolist())
                    else:
                        values.append(float(val))

                rows.extend(parents_indexes)
                cols.extend([v_index] * len(parents_indexes))

            alpha[cid] = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float64)

            i += 1
            if c_count >= 10 and i % (c_count // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // c_count)

        return alpha
    except:
        logger.error(traceback.format_exc())
        raise


def calc_beta(sequences, graph, k, r, user_map):
    try:
        u_count = len(user_map)
        c_count = len(sequences)
        beta = {}
        i = 0

        for cid, seq in sequences.items():
            values = []
            rows = []
            cols = []

            for v in seq.users:
                if v not in user_map:
                    continue
                v_index = user_map[v]
                parents = seq.get_active_parents(v, graph)
                if not parents:
                    continue
                parents_indexes = [user_map[uid] for uid in parents]
                parents_times = np.array([seq.user_times[vid] for vid in parents])
                user_time = np.ones(len(parents)) * seq.user_times[v]
                diff = (user_time - parents_times).reshape(len(parents), 1)
                diff[diff == 0] = 1.0 / (30 * 24 * 60)  # 1 minute
                k_col = k[parents_indexes, v_index].toarray()
                r_col = r[parents_indexes, v_index].toarray()
                k_exp = np.multiply(k_col, np.exp(-np.multiply(r_col, diff)))
                val = np.divide(k_exp, k_exp + 1 - k_col)
                if val.size > 1:
                    values.extend(np.squeeze(val).tolist())
                else:
                    values.append(float(val))
                rows.extend(parents_indexes)
                cols.extend([v_index] * len(parents_indexes))

            beta[cid] = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float64)

            i += 1
            if c_count >= 10 and i % (c_count // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // c_count)

        return beta
    except:
        logger.error(traceback.format_exc())
        raise


def calc_r(edges, alpha, beta, user_map, sequences, edge_pos_cascades):
    try:
        edges_count = len(edges)
        values = []
        rows = []
        cols = []
        i = 0
        logger.debug('len(alpha) = %s', len(alpha))
        logger.debugv('alpha = %s', alpha)

        for (u, v) in edges:

            u_index = user_map[u]
            v_index = user_map[v]
            pos_cascades = edge_pos_cascades[(u, v)]
            if pos_cascades:
                alphas = np.array([alpha[cid][u_index, v_index] for cid in pos_cascades])
                betas = np.array([beta[cid][u_index, v_index] for cid in pos_cascades])
                u_times = np.array([sequences[cid].user_times[u] for cid in pos_cascades])
                v_times = np.array([sequences[cid].user_times[v] for cid in pos_cascades])
                diff = v_times - u_times
                diff[diff == 0] = 1.0 / (30 * 24 * 60)  # 1 minute
                numerator = float(np.sum(alphas))
                denominator = float(np.sum(np.multiply(alphas + np.multiply(1 - alphas, betas), diff)))
                val = numerator / denominator

                values.append(val)
                rows.append(u_index)
                cols.append(v_index)

            i += 1
            if edges_count >= 10 and i % (edges_count // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // edges_count)

        return values, rows, cols
    except:
        logger.error(traceback.format_exc())
        raise


def calc_k(edges, alpha, beta, user_map, edge_pos_cascades, edge_neg_counts):
    try:
        edges_count = len(edges)
        values = []
        rows = []
        cols = []
        i = 0

        for (u, v) in edges:

            u_index = user_map[u]
            v_index = user_map[v]
            pos_cascades = edge_pos_cascades[(u, v)]

            if pos_cascades:
                neg_count = edge_neg_counts[(u, v)]
                alphas = np.array([alpha[cid][u_index, v_index] for cid in pos_cascades])
                betas = np.array([beta[cid][u_index, v_index] for cid in pos_cascades])
                val = np.sum(alphas + np.multiply(1 - alphas, betas)) / (len(pos_cascades) + neg_count)
                values.append(float(val))
                rows.append(u_index)
                cols.append(v_index)

            i += 1
            if edges_count >= 10 and i % (edges_count // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // edges_count)

        return values, rows, cols
    except:
        logger.error(traceback.format_exc())
        raise


class CTIC(IC):
    method = Method.CTIC
    max_iterations = 20
    stop_criterion = 1e-5

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold)
        self.k_param_name = 'k-ctic'
        self.r_param_name = 'r-ctic'

    def calc_parameters(self, train_set, project, multi_processed, eco, iterations=None, **kwargs):
        if iterations is None:
            iterations = self.max_iterations

        sequences = project.load_or_extract_act_seq(train_set)

        # Create maps from users and cascades db id's to their matrix id's.
        logger.debug('creating user and cascade id maps ...')
        user_ids = sorted(self.graph.nodes())
        user_map = {user_ids[i]: i for i in range(len(user_ids))}
        logger.info('train set size = %d', len(train_set))
        logger.info('user space size = %d', len(user_map))

        logger.info('extracting positive and negative examples ...')
        edge_pos_cascades, edge_neg_counts = self.__get_pos_neg_examples(sequences, self.graph, multi_processed)

        # Set initial values of k and r.
        k, r = self.__set_initial_values(self.graph, user_ids, user_map)

        # Run EM algorithm.
        logger.info('running algorithm ...')
        for i in range(iterations):
            with Timer('iteration time', level='debug'):
                logger.info('#%d' % (i + 1))

                logger.debug('calculating alpha ...')
                alpha = self.__calc_alpha_mp(sequences, self.graph, k, r, train_set, user_map, multi_processed)
                logger.debug('calculating beta ...')
                beta = self.__calc_beta_mp(sequences, self.graph, k, r, train_set, user_map, multi_processed)
                logger.debug('estimating r ...')
                last_r = r
                r = self.__calc_r_mp(sequences, self.graph, alpha, beta, user_map, edge_pos_cascades, multi_processed)
                logger.debug('estimating k ...')
                last_k = k
                k = self.__calc_k_mp(self.graph, alpha, beta, user_map, edge_pos_cascades, edge_neg_counts,
                                     multi_processed)

                if eco:
                    # Save r and k.
                    project.save_param(r, self.r_param_name, ParamTypes.SPARSE)
                    project.save_param(k, self.k_param_name, ParamTypes.SPARSE)

                # Calculate and report delta r and delta k.
                r_dif = r - last_r
                r_dif = np.sqrt(r_dif.multiply(r_dif).sum())
                k_dif = k - last_k
                k_dif = np.sqrt(k_dif.multiply(k_dif).sum())
                logger.info('r dif = %s, k dif = %s' % (r_dif, k_dif))
                logger.debug('r nnz = %d, k nnz = %d' % (r.nnz, k.nnz))
                del last_r
                del last_k

                if k_dif + r_dif < self.stop_criterion:
                    logger.info('Stop condition met: r dif + k dif < 1e-6')
                    break

        self.k = k.tocsr()
        self.r = r.tocsr()

    @time_measure('debug')
    def __set_initial_values(self, graph, user_ids, user_map):
        u_count = len(user_ids)
        k_values = []
        r_values = []
        rows = []
        cols = []
        i = 0
        for u_i in range(u_count):
            u = user_ids[u_i]
            children = list(graph.successors(u))
            if children:
                ch_indexes = [user_map[vid] for vid in children]
                k_values.extend([1 / len(children)] * len(children))
                r_values.extend([1 / 30] * len(children))  # approximately 1 day
                rows.extend([u_i] * len(children))
                cols.extend(ch_indexes)
            i += 1
            if i % (u_count // 10) == 0:
                logger.debug('%d%% done' % (i * 100 // u_count))

        k = sparse.csc_matrix((k_values, [rows, cols]), shape=(u_count, u_count), dtype=np.float32)
        r = sparse.csc_matrix((r_values, [rows, cols]), shape=(u_count, u_count), dtype=np.float32)
        return k, r

    @time_measure('debug')
    def __calc_a_mp(self, sequences, graph, k, r, cascade_ids, user_map, multi_processed):
        c_count = len(cascade_ids)

        if multi_processed and settings.TRAIN_WORKERS > 1:
            process_count = min(settings.TRAIN_WORKERS, c_count)
            pool = Pool(processes=process_count)
            step = int(math.ceil(c_count / process_count))
            results = []
            for j in range(0, c_count, step):
                subset = cascade_ids[j: j + step]
                sequences_j = {cid: sequences[cid] for cid in subset}
                res = pool.apply_async(calc_a, (sequences_j, graph, k, r, user_map))
                results.append(res)

            pool.close()
            pool.join()

            # Collect results of the processes.
            a = {}
            for i in range(len(results)):
                a_subset = results[i].get()
                a.update(a_subset)
        else:
            a = calc_a(sequences, graph, k, r, user_map)

        return a

    @time_measure('debug')
    def __calc_b_mp(self, sequences, graph, k, r, cascade_ids, user_map, multi_processed):
        c_count = len(cascade_ids)

        if multi_processed and settings.TRAIN_WORKERS > 1:
            process_count = min(settings.TRAIN_WORKERS, c_count)
            pool = Pool(processes=process_count)
            step = int(math.ceil(c_count / process_count))
            results = []
            for j in range(0, c_count, step):
                subset = cascade_ids[j: j + step]
                sequences_j = {cid: sequences[cid] for cid in subset}
                res = pool.apply_async(calc_b, (sequences_j, graph, k, r, user_map))
                results.append(res)

            pool.close()
            pool.join()

            # Collect results of the processes.
            b = {}
            for i in range(len(results)):
                a_subset = results[i].get()
                b.update(a_subset)
        else:
            b = calc_b(sequences, graph, k, r, user_map)

        return b

    @time_measure('debug')
    def __calc_alpha_mp(self, sequences, graph, k, r, cascade_ids, user_map, multi_processed):
        c_count = len(cascade_ids)

        if multi_processed and settings.TRAIN_WORKERS > 1:
            step = 500
            sequences_parts = [{cid: sequences[cid] for cid in cascade_ids[i:i + step]} for i in
                               range(0, c_count, step)]
            with ProcessPoolExecutor(max_workers=settings.TRAIN_WORKERS) as executor:
                alpha_subsets = executor.map(calc_alpha, sequences_parts, repeat(graph), repeat(k), repeat(r),
                                             repeat(user_map))
            del sequences_parts

            # Collect results of the processes.
            logger.debug('merging results of processes ...')
            alpha = {}
            for a_subset in alpha_subsets:
                alpha.update(a_subset)
                a_subset.clear()
        else:
            alpha = calc_alpha(sequences, graph, k, r, user_map)

        return alpha

    @time_measure('debug')
    def __calc_beta_mp(self, sequences, graph, k, r, cascade_ids, user_map, multi_processed):
        c_count = len(cascade_ids)

        if multi_processed and settings.TRAIN_WORKERS > 1:
            process_count = min(settings.TRAIN_WORKERS, c_count)
            pool = Pool(processes=process_count)
            step = int(math.ceil(c_count / process_count))
            results = []
            for j in range(0, c_count, step):
                subset = cascade_ids[j: j + step]
                sequences_j = {cid: sequences[cid] for cid in subset}
                res = pool.apply_async(calc_beta, (sequences_j, graph, k, r, user_map))
                results.append(res)

            pool.close()
            pool.join()

            # Collect results of the processes.
            beta = {}
            for i in range(len(results)):
                a_subset = results[i].get()
                beta.update(a_subset)
        else:
            beta = calc_beta(sequences, graph, k, r, user_map)

        return beta

    @time_measure('debug')
    def __calc_r_mp(self, sequences, graph, alpha, beta, user_map, edge_pos_cascades, multi_processed):
        edges = list(graph.edges())
        e_count = len(edges)
        u_count = len(user_map)

        multi_processed = False  # TODO: Remove it!

        if multi_processed and settings.TRAIN_WORKERS > 1:

            step = e_count // settings.TRAIN_WORKERS
            step = max(10, min(10000, step))
            logger.info('creating %d jobs to calculate r ...', math.ceil(e_count / step))

            with ProcessPoolExecutor(max_workers=settings.TRAIN_WORKERS) as executor:
                edge_lists = [edges[j: j + step] for j in range(0, e_count, step)]
                results = executor.map(calc_r, edge_lists, repeat(alpha), repeat(beta), repeat(user_map),
                                       repeat(sequences), repeat(edge_pos_cascades))

            # Collect results of the processes.
            values = []
            rows = []
            cols = []
            for val_subset, row_subset, col_subset in results:
                values.extend(val_subset)
                rows.extend(row_subset)
                cols.extend(col_subset)
        else:
            values, rows, cols = calc_r(edges, alpha, beta, user_map, sequences, edge_pos_cascades)

        r = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float64)
        return r

    @time_measure('debug')
    def __calc_k_mp(self, graph, alpha, beta, user_map, edge_pos_cascades, edge_neg_counts, multi_processed):
        edges = list(graph.edges())
        e_count = len(edges)
        u_count = len(user_map)

        multi_processed = False  # TODO: Remove it!

        if multi_processed and settings.TRAIN_WORKERS > 1:
            process_count = min(settings.TRAIN_WORKERS, e_count)
            pool = Pool(processes=process_count)
            step = int(math.ceil(e_count / process_count))
            results = []
            for j in range(0, e_count, step):
                edges_j = edges[j: j + step]
                res = pool.apply_async(calc_k, (edges_j, alpha, beta, user_map, edge_pos_cascades, edge_neg_counts))
                results.append(res)

            pool.close()
            pool.join()

            # Collect results of the processes.
            values = []
            rows = []
            cols = []
            for res in results:
                val_subset, row_subset, col_subset = res.get()
                values.extend(val_subset)
                rows.extend(row_subset)
                cols.extend(col_subset)
        else:
            values, rows, cols = calc_k(edges, alpha, beta, user_map, edge_pos_cascades, edge_neg_counts)

        k = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float64)
        return k

    def __get_pos_neg_examples(self, sequences, graph, multi_processed):
        """
        :param sequences:
        :param graph:
        :return: tuple of (edge_pos_cascades, edge_neg_counts)
                edge_pos_cascades : dictionary of edge (u,v) to the cascade ids of positive examples of the edge.
                edge_neg_counts : dictionary of edge (u,v) to the number of negative examples of the edge.
        """

        if multi_processed and settings.DEFAULT_WORKERS > 1:
            e_count = graph.number_of_edges()
            step = e_count // settings.TRAIN_WORKERS
            step = max(10, min(10000, step))

            with ProcessPoolExecutor(max_workers=settings.DEFAULT_WORKERS) as executor:
                edges = list(graph.edges())
                edge_lists = [edges[j: j + step] for j in range(0, e_count, step)]
                del edges
                results = executor.map(get_pos_neg_examples, edge_lists, repeat(sequences))

            edge_pos_cascades, edge_neg_counts = {}, {}
            for pos_res, neg_res in results:
                edge_pos_cascades.update(pos_res)
                edge_neg_counts.update(neg_res)
        else:
            edge_pos_cascades, edge_neg_counts = get_pos_neg_examples(list(graph.edges()), sequences)

        return edge_pos_cascades, edge_neg_counts


def get_pos_neg_examples(edges_list: list, sequences: dict):
    edge_pos_cascades = {(u, v): [cid for cid, seq in sequences.items() if
                                  v in seq.user_times and u in seq.user_times and seq.user_times[u] <=
                                  seq.user_times[v]] for (u, v) in edges_list}
    edge_neg_counts = {
        (u, v): sum(1 for cid, seq in sequences.items() if v not in seq.user_times and u in seq.user_times) for
        (u, v) in edges_list}
    return edge_pos_cascades, edge_neg_counts
