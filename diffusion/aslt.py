import math
import time
import traceback
from functools import reduce
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

import settings
from cascade.models import ParamTypes
from diffusion.enum import Method
from diffusion.models import LT
from settings import logger
from utils.time_utils import Timer, time_measure


def calc_h(sequences, graph, w, r, user_map):
    try:
        c_count = len(sequences)
        values = []
        rows = []
        cols = []

        i = 0
        for cindex, sequence in sequences.items():
            for uid in sequence.users:
                if str(uid) not in user_map:
                    continue
                uindex = user_map[str(uid)]
                val = 0
                if sequence.user_times[uid] == sequence.times[0]:
                    val = 1
                else:
                    active_parents = sequence.get_active_parents(uid, graph)
                    if active_parents:
                        p_count = len(active_parents)
                        act_par_indexes = [user_map[str(id)] for id in active_parents]
                        act_par_times = np.array([sequence.user_times[pid] for pid in active_parents])
                        user_time = np.ones(p_count) * sequence.user_times[uid]
                        diff = user_time - act_par_times
                        diff[diff == 0] = 1.0 / (30 * 24 * 60)  # 1 minute
                        w_col = w[:, uindex].toarray()
                        val = np.exp(-r[uindex] * diff).dot(w_col[act_par_indexes]) * r[uindex]

                        if val == 0:
                            logger.warning('\th = 0')

                if val:
                    values.append(val)
                    rows.append(cindex)
                    cols.append(uindex)
            i += 1
            if c_count >= 10 and i % (c_count // 10) == 0:
                logger.debug('\t%d%% done' % (i * 100 // c_count))

        return values, rows, cols
    except:
        logger.error(traceback.format_exc())
        raise


def calc_g(sequences, graph, w, r, user_map):
    try:
        c_count = len(sequences)
        values = []
        rows = []
        cols = []
        i = 0

        for cindex, sequence in sequences.items():
            rond_set = sequence.get_rond_set(graph)

            for uid in rond_set:
                uindex = user_map[str(uid)]
                active_parents = sequence.get_active_parents(uid, graph)
                act_par_indexes = [user_map[str(id)] for id in active_parents]
                inactive_parents = set(graph.predecessors(uid)) - set(active_parents)
                inact_par_indexes = [user_map[str(id)] for id in inactive_parents]

                w_col = w[:, uindex].todense()
                inact_par_sum = w_col[inact_par_indexes].sum()
                act_par_times = np.matrix([[sequence.user_times[pid] for pid in active_parents]])
                max_time = np.repeat(np.matrix(sequence.max_t), len(active_parents))
                diff = max_time - act_par_times
                act_par_sum = np.exp(-r[uindex] * diff) * w_col[act_par_indexes]

                val = w[uindex, uindex] + inact_par_sum + float(act_par_sum)
                if np.float32(val) == 0:
                    logger.warning('\tg = 0')
                values.append(val)
                rows.append(cindex)
                cols.append(uindex)

            i += 1
            if c_count >= 10 and i % (c_count // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // c_count)

        return values, rows, cols
    except:
        logger.error(traceback.format_exc())
        raise


def calc_phi_h(sequences, graph, w, r, h, user_map):
    try:
        u_count = len(user_map)
        c_count = len(sequences)
        phi_h = {}
        i = 0

        for cindex, sequence in sequences.items():
            values = []
            rows = []
            cols = []

            for v in sequence.users:
                if str(v) not in user_map:
                    continue
                vindex = user_map[str(v)]
                active_parents = sequence.get_active_parents(v, graph)
                if not active_parents:
                    continue
                act_par_indexes = [user_map[str(id)] for id in active_parents]
                act_par_times = np.matrix([[sequence.user_times[pid] for pid in active_parents]])
                user_time = np.repeat(np.matrix([sequence.user_times[v]]), len(active_parents))
                diff = user_time - act_par_times
                diff[diff == 0] = 1.0 / (30 * 24 * 60)  # 1 minute
                w_col = w[:, vindex].todense()
                val = np.multiply(w_col[act_par_indexes].T, np.exp(-r[vindex] * diff)) * r[vindex] / h[cindex, vindex]
                if np.isinf(np.float32(val)).any():
                    logger.warning('\tphi_h = inf')
                    # if (np.float32(val) == 0).any():
                #    logger.warning('\phi_h = 0')
                if val.size > 1:
                    values.extend(list(np.array(val).squeeze()))
                else:
                    values.append(float(val))
                rows.extend(act_par_indexes)
                cols.extend([vindex] * len(act_par_indexes))

            phi_h[cindex] = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float32)

            i += 1
            if c_count >= 10 and i % (c_count // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // c_count)

        # logger.debug('size of phi_h subset: %f G', asizeof(phi_h) / 1024 ** 3)
        return phi_h
    except:
        logger.error(traceback.format_exc())
        raise


def calc_phi_g(sequences, graph, w, g, user_map):
    try:
        u_count = len(user_map)
        c_count = len(sequences)
        phi_g = {}
        i = 0

        for cindex, sequence in sequences.items():
            values = []
            rows = []
            cols = []

            for v in sequence.get_rond_set(graph):
                v_i = user_map[str(v)]
                u_set = {v} | (set(graph.predecessors(v)) - set(sequence.get_active_parents(v, graph)))
                if not u_set:
                    continue
                u_indexes = [user_map[str(id)] for id in u_set]
                w_col = w[:, v_i].todense()
                val = w_col[u_indexes] / g[cindex, v_i]
                if np.isinf(np.float32(val)).any():
                    logger.warning('\tphi_g = inf')
                    # if (np.float32(val) == 0).any():
                #    logger.warning('\phi_g = 0')
                if val.size > 1:
                    values.extend(list(np.array(val).squeeze()))
                else:
                    values.append(float(val))
                rows.extend(u_indexes)
                cols.extend([v_i] * len(u_indexes))

            phi_g[cindex] = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float32)

            i += 1
            if c_count >= 10 and i % (c_count // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // c_count)

        # logger.debug('size of phi_g subset: %f G', asizeof(phi_g) / 1024 ** 3)
        return phi_g
    except:
        logger.error(traceback.format_exc())
        raise


def calc_psi(sequences, graph, w, r, g, user_map):
    try:
        u_count = len(user_map)
        c_count = len(sequences)
        psi = {}
        i = 0

        for cindex, sequence in sequences.items():
            values = []
            rows = []
            cols = []

            for v in sequence.get_rond_set(graph):
                v_i = user_map[str(v)]
                active_parents = sequence.get_active_parents(v, graph)
                act_par_indexes = [user_map[str(id)] for id in active_parents]
                act_par_times = np.matrix([[sequence.user_times[pid] for pid in active_parents]])
                max_time = np.repeat(np.matrix([sequence.max_t]), len(active_parents))
                diff = max_time - act_par_times
                w_col = w[:, v_i].todense()
                val = np.multiply(w_col[act_par_indexes].T, np.exp(-r[v_i] * diff)) / g[cindex, v_i]
                if np.isinf(np.float32(val)).any():
                    logger.warning('\tpsi = inf')
                    # if (np.float32(val) == 0).any():
                #    logger.warning('\psi = 0')
                if val.size > 1:
                    values.extend(list(np.array(val).squeeze()))
                else:
                    values.append(float(val))
                rows.extend(act_par_indexes)
                cols.extend([v_i] * len(act_par_indexes))

            psi[cindex] = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float32)

            i += 1
            if c_count >= 10 and i % (c_count // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // c_count)

        # logger.debug('size of psi subset: %f G', asizeof(psi) / 1024 ** 3)
        return psi
    except:
        logger.error(traceback.format_exc())
        raise


def calc_r(sequences, graph, phi_h, psi, user_ids, cascade_map, user_map, c_set1, c_set2):
    try:
        r_values = []

        logger.debug('\tcalculating values ...')
        i = 0
        t0 = time.time()

        for v in user_ids:
            v_i = user_map[str(v)]

            phi_sum = 0
            phi_time_sum = 0
            psi_time_sum = 0
            for c in set(c_set1[v]) | set(c_set2[v]):
                c_i = cascade_map[str(c)]
                act_seq = sequences[c]
                active_parents = act_seq.get_active_parents(v, graph)
                if not active_parents:
                    continue
                act_par_indexes = [user_map[str(id)] for id in active_parents]
                act_par_times = np.matrix([[act_seq.user_times[pid] for pid in active_parents]])

                if c in c_set1[v]:
                    phi_h_col = phi_h[c_i][:, v_i].todense()
                    phi_sum += float(phi_h_col[act_par_indexes].sum())
                    user_time = np.repeat(np.matrix([act_seq.user_times[v]]), len(active_parents))
                    diff = user_time - act_par_times
                    diff[diff == 0] = 1.0 / (24 * 60)  # 1 hour
                    phi_time_sum += float(diff * phi_h_col[act_par_indexes])

                if c in c_set2[v]:
                    psi_col = psi[c_i][:, v_i]
                    max_time = np.repeat(np.matrix([act_seq.max_t]), len(active_parents))
                    diff = max_time - act_par_times
                    psi_time_sum += float(diff * psi_col[act_par_indexes])

            if phi_sum == 0:
                r_values.append(0)
                # if m_set1[v] or m_set2[v]:
                #    logger.info('\tWARNING: r = 0, sets: %s, %s' % (m_set1[v], m_set2[v]))
            else:
                if phi_time_sum + psi_time_sum != 0:
                    r_values.append(phi_sum / (phi_time_sum + psi_time_sum))
                else:
                    r_values.append(np.finfo(np.float32).max)
                    logger.warning('\tdenominator = 0, r = inf')

            i += 1
            if time.time() - t0 > 60:
                logger.debug('\t%d from %d r values (%d%%) done', i, len(user_ids), i * 100 // len(user_ids))
                t0 = time.time()

        logger.debug('\tdone')
        return r_values
    except:
        logger.error(traceback.format_exc())
        raise


class AsLT(LT):
    method = Method.ASLT
    max_iterations = 20

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold)
        self.w_param_name = 'w-aslt'
        self.r_param_name = 'r-aslt'

    def calc_parameters(self, train_set, project, multi_processed, eco, iterations=None, **kwargs):
        if iterations is None:
            iterations = self.max_iterations

        graph, sequences = project.load_or_extract_graph_seq(train_set)

        # Create maps from users and cascades db id's to their matrix id's.
        logger.debug('creating user and cascade id maps ...')
        user_ids = sorted(graph.nodes())
        user_map = {str(user_ids[i]): i for i in range(len(user_ids))}
        cascade_map = {str(train_set[i]): i for i in range(len(train_set))}
        logger.info('train set size = %d', len(train_set))
        logger.info('user space size = %d', len(user_map))

        # Set initial values of w and r.
        w, r = self.__initialize(user_ids, user_map, graph, project, eco)

        # Run EM algorithm.
        logger.info('running algorithm ...')
        for i in range(iterations):
            with Timer('iteration time'):
                logger.info('#%d' % (i + 1))

                h = self._load_or_calc_h(sequences, graph, w, r, train_set, cascade_map, user_map, project,
                                         multi_processed, eco)
                phi_h = self.__load_or_calc_phi_h(sequences, graph, w, r, h, train_set, cascade_map, user_map, project,
                                                  multi_processed, eco)
                g = self.__load_or_calc_g(sequences, graph, w, r, train_set, cascade_map, user_map, project,
                                          multi_processed,
                                          eco)
                phi_g = self.__load_or_calc_phi_g(sequences, graph, w, g, train_set, cascade_map, user_map, project,
                                                  multi_processed, eco)
                psi = self.__load_or_calc_psi(sequences, graph, w, r, g, train_set, cascade_map, user_map, project,
                                              multi_processed, eco)
                del g

                logger.debug('estimating r ...')
                last_r = r
                r_multi_processed = multi_processed and len(user_ids) < 2 * 10 ** 7
                r = self.__calc_r_mp(sequences, graph, phi_h, psi, user_ids, train_set, cascade_map, user_map,
                                     r_multi_processed)

                if eco:
                    phi_g = project.load_param('phi_g', ParamTypes.SPARSE_LIST)
                    logger.debug('phi_g loaded')

                logger.debug('estimating w ...')
                last_w = w
                w = self.__calc_w(sequences, graph, phi_h, phi_g, psi, user_ids, user_map, train_set, cascade_map)

                del phi_h
                del phi_g
                del psi

                # Calculate and report delta r and delta w.
                r_dif = np.linalg.norm(r - last_r)
                w_dif = w - last_w
                w_dif = np.sqrt(w_dif.multiply(w_dif).sum())
                logger.info('r dif = %s, w dif = %s' % (r_dif, w_dif))
                logger.info('r nnz = %d, w nnz = %d' % (np.count_nonzero(r), w.nnz))
                del last_r
                del last_w

                if w_dif + r_dif < 1e-6:
                    logger.info('Stop condition met: r dif + w dif < 1e-6')
                    break

        if eco:
            # Save r and w.
            project.save_param(r, self.r_param_name, ParamTypes.ARRAY)
            project.save_param(w, self.w_param_name, ParamTypes.SPARSE)

        self.w = w.toarray()
        self.r = r

    def __load_or_calc_psi(self, sequences, graph, w, r, g, train_set, cascade_map, user_map, project, multi_processed,
                           eco):
        data_loaded = False
        if eco:
            try:
                psi = project.load_param('psi', ParamTypes.SPARSE_LIST)
                logger.debug('psi loaded')
                data_loaded = True
            except FileNotFoundError:
                pass
        if not data_loaded:
            logger.debug('calculating psi ...')
            psi = self.__calc_psi_mp(sequences, graph, w, r, g, train_set, cascade_map, user_map, multi_processed)
        return psi

    def __load_or_calc_phi_g(self, sequences, graph, w, g, train_set, cascade_map, user_map, project, multi_processed,
                             eco):
        data_loaded = False
        if eco:
            try:
                phi_g = project.load_param('phi_g', ParamTypes.SPARSE_LIST)  # Just check if exists.
                data_loaded = True
            except FileNotFoundError:
                pass
        if not data_loaded:
            logger.debug('calculating phi_g ...')
            phi_g = self.__calc_phi_g_mp(sequences, graph, w, g, train_set, cascade_map, user_map, multi_processed)
        return phi_g

    def __load_or_calc_phi_h(self, sequences, graph, w, r, h, train_set, cascade_map, user_map, project,
                             multi_processed, eco):
        data_loaded = False
        if eco:
            try:
                phi_h = project.load_param('phi_h', ParamTypes.SPARSE_LIST)  # Just check if exists.
                data_loaded = True
            except FileNotFoundError:
                pass
        if not data_loaded:
            logger.debug('calculating phi_h ...')
            phi_h = self.__calc_phi_h_mp(sequences, graph, w, r, h, train_set, cascade_map, user_map, multi_processed)
        return phi_h

    def __load_or_calc_g(self, sequences, graph, w, r, train_set, cascade_map, user_map, project, multi_processed, eco):
        data_loaded = False
        if eco:
            try:
                g = project.load_param('g', ParamTypes.SPARSE)
                logger.debug('g loaded')
                data_loaded = True
            except FileNotFoundError:
                pass
        if not data_loaded:
            logger.debug('calculating g ...')
            g = self.__calc_g_mp(sequences, graph, w, r, train_set, cascade_map, user_map, multi_processed)
        return g

    def _load_or_calc_h(self, sequences, graph, w, r, train_set, cascade_map, user_map, project, multi_processed, eco):
        data_loaded = False
        if eco:
            try:
                h = project.load_param('h', ParamTypes.SPARSE)
                logger.debug('h loaded')
                data_loaded = True
            except FileNotFoundError:
                pass
        if not data_loaded:
            logger.debug('calculating h ...')
            h = self.__calc_h_mp(sequences, graph, w, r, train_set, cascade_map, user_map, multi_processed)
        return h

    def __initialize(self, user_ids, user_map, graph, project, eco):
        data_loaded = False
        if eco:
            try:
                w = project.load_param(self.w_param_name, ParamTypes.SPARSE)
                r = project.load_param(self.r_param_name, ParamTypes.ARRAY)
                logger.debug('w and r loaded')
                data_loaded = True
            except FileNotFoundError:
                pass
        if not data_loaded:
            logger.debug('initializing parameters ...')
            w, r = self.__set_initial_values(graph, user_ids, user_map)
        return w, r

    @time_measure('debug')
    def __set_initial_values(self, graph, user_ids, user_map):
        u_count = len(user_ids)
        nodes = graph.nodes()
        values = []
        rows = []
        cols = []
        i = 0
        for v_i in range(u_count):
            v = user_ids[v_i]
            if v in nodes:
                parents = list(graph.predecessors(v))
                parents.append(v)
                par_indexes = [user_map[str(uid)] for uid in parents]
                values.extend([1.0 / len(parents)] * len(parents))
                rows.extend(par_indexes)
                cols.extend([v_i] * len(parents))
            else:
                values.append(1)
                rows.append(v_i)
                cols.append(v_i)
            i += 1
            if i % (u_count // 10) == 0:
                logger.debug('%d%% done' % (i * 100 // u_count))

        w = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float32)
        r = np.ones(u_count, np.float32) / 30  # approximately 1 day
        return w, r

    @time_measure('debug')
    def __calc_h_mp(self, data, graph, w, r, cascade_ids, cascade_map, user_map, multi_processed):
        u_count = len(user_map)
        c_count = len(cascade_ids)

        if multi_processed:
            process_count = min(settings.TRAIN_WORKERS, c_count)
            pool = Pool(processes=process_count)
            step = int(math.ceil(c_count / process_count))
            results = []
            for j in range(0, c_count, step):
                subset = cascade_ids[j: j + step]
                sequences = {cascade_map[str(cid)]: data[cid] for cid in subset}
                res = pool.apply_async(calc_h, (sequences, graph, w, r, user_map))
                results.append(res)

            pool.close()
            pool.join()

            # Collect results of the processes.
            values = []
            rows = []
            cols = []
            for i in range(len(results)):
                val_subset, row_subset, col_subset = results[i].get()
                values.extend(val_subset)
                rows.extend(row_subset)
                cols.extend(col_subset)
        else:
            sequences = {cascade_map[str(cid)]: data[cid] for cid in data}
            values, rows, cols = calc_h(sequences, graph, w, r, user_map)

        values = np.array(values, dtype=np.float64)
        h = sparse.csc_matrix((values, [rows, cols]), shape=(c_count, u_count), dtype=np.float64)
        logger.debug('number of zero h: %d', np.count_nonzero(h.data == 0))
        return h

    @time_measure('debug')
    def __calc_g_mp(self, data, graph, w, r, cascade_ids, cascade_map, user_map, multi_processed):
        u_count = len(user_map)
        c_count = len(cascade_ids)

        if multi_processed:
            process_count = min(settings.TRAIN_WORKERS, c_count)
            pool = Pool(processes=process_count)
            step = int(math.ceil(c_count / process_count))
            results = []
            for j in range(0, c_count, step):
                subset = cascade_ids[j: j + step]
                sequences = {cascade_map[str(cid)]: data[cid] for cid in subset}
                res = pool.apply_async(calc_g, (sequences, graph, w, r, user_map))
                results.append(res)

            pool.close()
            pool.join()

            # Collect results of the processes.
            values = []
            rows = []
            cols = []
            for i in range(len(results)):
                val_subset, row_subset, col_subset = results[i].get()
                values.extend(val_subset)
                rows.extend(row_subset)
                cols.extend(col_subset)
        else:
            sequences = {cascade_map[str(cid)]: data[cid] for cid in data}
            values, rows, cols = calc_g(sequences, graph, w, r, user_map)

        g = sparse.csc_matrix((values, [rows, cols]), shape=(c_count, u_count), dtype=np.float32)
        return g

    @time_measure('debug')
    def __calc_phi_h_mp(self, data, graph, w, r, h, cascade_ids, cascade_map, user_map, multi_processed):
        c_count = len(cascade_ids)

        if multi_processed:
            process_count = min(settings.TRAIN_WORKERS, c_count)
            pool = Pool(processes=process_count)
            step = int(math.ceil(c_count / process_count))
            results = []
            for j in range(0, c_count, step):
                subset = cascade_ids[j: j + step]
                sequences = {cascade_map[str(cid)]: data[cid] for cid in subset}
                res = pool.apply_async(calc_phi_h, (sequences, graph, w, r, h, user_map))
                results.append(res)

            pool.close()
            pool.join()

            # Collect results of the processes.
            phi_h_dict = {}
            for i in range(len(results)):
                phi_h_subset = results[i].get()
                phi_h_dict.update(phi_h_subset)
        else:
            sequences = {cascade_map[str(cid)]: data[cid] for cid in data}
            phi_h_dict = calc_phi_h(sequences, graph, w, r, h, user_map)

        phi_h = [None for _ in range(len(cascade_ids))]
        for cindex, mat in phi_h_dict.items():
            phi_h[cindex] = mat
        return phi_h

    @time_measure('debug')
    def __calc_phi_g_mp(self, data, graph, w, g, cascade_ids, cascade_map, user_map, multi_processed):
        c_count = len(cascade_ids)

        if multi_processed:
            process_count = min(settings.TRAIN_WORKERS, c_count)
            pool = Pool(processes=process_count)
            step = int(math.ceil(c_count / process_count))
            results = []
            for j in range(0, c_count, step):
                subset = cascade_ids[j: j + step]
                sequences = {cascade_map[str(cid)]: data[cid] for cid in subset}
                res = pool.apply_async(calc_phi_g, (sequences, graph, w, g, user_map))
                results.append(res)

            pool.close()
            pool.join()

            # Collect results of the processes.
            phi_g_dict = {}
            for i in range(len(results)):
                phi_g_subset = results[i].get()
                phi_g_dict.update(phi_g_subset)
        else:
            sequences = {cascade_map[str(cid)]: data[cid] for cid in data}
            phi_g_dict = calc_phi_g(sequences, graph, w, g, user_map)

        phi_g = [None for _ in range(len(cascade_ids))]
        for cindex, mat in phi_g_dict.items():
            phi_g[cindex] = mat

        return phi_g

    @time_measure('debug')
    def __calc_psi_mp(self, data, graph, w, r, g, cascade_ids, cascade_map, user_map, multi_processed):
        c_count = len(cascade_ids)

        if multi_processed:
            process_count = min(settings.TRAIN_WORKERS, c_count)
            pool = Pool(processes=process_count)
            step = int(math.ceil(c_count / process_count))
            results = []
            for j in range(0, c_count, step):
                subset = cascade_ids[j: j + step]
                sequences = {cascade_map[str(cid)]: data[cid] for cid in subset}
                res = pool.apply_async(calc_psi, (sequences, graph, w, r, g, user_map))
                results.append(res)

            pool.close()
            pool.join()

            # Collect results of the processes.
            psi_dict = {}
            for i in range(len(results)):
                psi_subset = results[i].get()
                psi_dict.update(psi_subset)
        else:
            sequences = {cascade_map[str(cid)]: data[cid] for cid in data}
            psi_dict = calc_psi(sequences, graph, w, r, g, user_map)

        psi = [None for _ in range(c_count)]
        for cindex, mat in psi_dict.items():
            psi[cindex] = mat

        return psi

    @time_measure('debug')
    def __calc_r_mp(self, data, graph, phi_h, psi, user_ids, cascade_ids, cascade_map, user_map, multi_processed):
        u_count = len(user_ids)
        c_count = len(cascade_ids)

        logger.debug('\textracting sigma domains ...')
        c_set1 = {v: [] for v in user_ids}
        c_set2 = {v: [] for v in user_ids}
        uid_set = set(user_ids)
        for c in cascade_ids:
            for v in set(data[c].users) & uid_set:
                c_set1[v].append(c)
            for v in data[c].get_rond_set(graph) & uid_set:
                c_set2[v].append(c)

        if multi_processed:
            logger.debug('\tcreating processes to calculate values ...')
            process_count = min(settings.TRAIN_WORKERS, u_count)
            pool = Pool(processes=process_count)
            step = int(math.ceil(u_count / process_count))
            results = []
            for j in range(0, u_count, step):
                subset = user_ids[j: j + step]
                c_set1_subset = {v: c_set1.pop(v) for v in subset}
                c_set2_subset = {v: c_set2.pop(v) for v in subset}
                # Get union of all c_set1 and c_set2 values of users in the subset.
                related_cascades = reduce(lambda x, y: x | set(y), list(c_set1_subset.values()), set())
                related_cascades.update(reduce(lambda x, y: x | set(y), list(c_set2_subset.values()), set()))
                # Get phi_h and psi of just the related cascades.
                related_indexes = set(cascade_map[str(cid)] for cid in related_cascades)
                cur_phi_h = {i: phi_h[i] for i in range(c_count) if i in related_indexes}
                cur_psi = {i: psi[i] for i in range(c_count) if i in related_indexes}
                res = pool.apply_async(calc_r, (
                    data, graph, cur_phi_h, cur_psi, subset, cascade_map, user_map, c_set1_subset, c_set2_subset))
                results.append(res)

            del subset, c_set1, c_set2, c_set1_subset, c_set2_subset, related_cascades, cur_phi_h, cur_psi
            pool.close()
            pool.join()

            # Collect results of the processes.
            r_values = []
            for res in results:
                r_values.extend(res.get())
        else:
            r_values = calc_r(data, graph, phi_h, psi, user_ids, cascade_map, user_map, c_set1, c_set2)

        # Convert to numpy array.
        r = np.ones(u_count, np.float32)
        for i in range(u_count):
            r[user_map[str(user_ids[i])]] = r_values[i]

        return r

    @time_measure('debug')
    def __calc_w(self, data, graph, phi_h, phi_g, psi, user_ids, user_map, cascade_ids, cascade_map):
        u_count = len(user_ids)

        logger.debug('\textracting sigma domains ...')
        mv_set2 = {v: [] for v in graph.nodes()}
        muv_set1 = {edge: [] for edge in graph.edges()}
        muv_set2 = {edge: [] for edge in graph.edges()}
        muv_set3 = {edge: [] for edge in graph.edges()}
        for m in cascade_ids:
            cascade = data[m]
            for v in cascade.users:
                for u in cascade.get_active_parents(v, graph):
                    muv_set1[(u, v)].append(m)
            for v in cascade.get_rond_set(graph):
                mv_set2[v].append(m)
                for u in set(graph.predecessors(v)) - set(cascade.get_active_parents(v, graph)):
                    muv_set2[(u, v)].append(m)
                for u in cascade.get_active_parents(v, graph):
                    muv_set3[(u, v)].append(m)

        logger.debug('\tcalculating values ...')
        values = []
        rows = []
        cols = []
        val_count = len(graph.edges()) + len(graph.nodes())
        i = 0
        for (u, v) in graph.edges():
            u_i = user_map[str(u)]
            v_i = user_map[str(v)]
            phi_h_sum = 0
            phi_g_sum = 0
            psi_sum = 0
            if muv_set1[(u, v)]:
                phi_h_sum = np.array([phi_h[cascade_map[str(m)]][u_i, v_i] for m in muv_set1[(u, v)]]).sum()
            if muv_set2[(u, v)]:
                phi_g_sum = np.array([phi_g[cascade_map[str(m)]][u_i, v_i] for m in muv_set2[(u, v)]]).sum()
            if muv_set3[(u, v)]:
                psi_sum = np.array([psi[cascade_map[str(m)]][u_i, v_i] for m in muv_set3[(u, v)]]).sum()
            val = phi_h_sum + phi_g_sum + psi_sum
            if val:
                values.append(val)
                rows.append(u_i)
                cols.append(v_i)
                # elif muv_set1[(u, v)] or muv_set2[(u, v)] or muv_set3[(u, v)]:
            #    logger.warning('\w = 0 at %s, sets: %s, %s, %s' % (
            #        (u, v), muv_set1[(u, v)], muv_set2[(u, v)], muv_set3[(u, v)]))

            i += 1
            if val_count >= 10 and i % (val_count // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // val_count)

        for v in graph.nodes():
            v_i = user_map[str(v)]
            if mv_set2[v]:
                phi_g_sum = np.array([phi_g[cascade_map[str(m)]][v_i, v_i] for m in mv_set2[v]]).sum()
                if phi_g_sum:
                    values.append(phi_g_sum)
                    rows.append(v_i)
                    cols.append(v_i)
                    # else:
                    #    logger.warning('\t\tw = 0 at %s, set: %s' % ((v, v), mv_set2[v]))

            i += 1
            if val_count >= 10 and i % (val_count // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // val_count)

        w = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype='d')

        logger.debug('\tnormalizing w ...')
        w = normalize(w, axis=0, copy=False)
        w = sparse.csc_matrix((w.data, w.indices, w.indptr), shape=w.shape, dtype=np.float32)

        return w
