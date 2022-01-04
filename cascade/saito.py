import math
import traceback
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

import settings
from cascade.models import AsLT, ParamTypes
from db.managers import DBManager
from settings import logger
from utils.time_utils import Timer, time_measure


def calc_h(sequences, graph, w, r, user_map):
    try:
        c_count = len(sequences)
        values = []
        rows = []
        cols = []

        i = 0
        for mindex, sequence in sequences.items():
            for uid in sequence.users:
                uindex = user_map[str(uid)]
                val = 0
                if sequence.user_times[uid] == sequence.times[0]:
                    val = 1
                else:
                    active_parents = sequence.get_active_parents(uid, graph)
                    if active_parents:
                        act_par_indexes = [user_map[str(id)] for id in active_parents]
                        act_par_times = np.matrix([[sequence.user_times[pid] for pid in active_parents]])
                        user_time = np.repeat(np.matrix([sequence.user_times[uid]]), len(active_parents))
                        diff = user_time - act_par_times
                        diff[diff == 0] = 1.0 / (24 * 60)  # 1 minute
                        w_col = w[:, uindex].todense()
                        val = float(np.exp(-r[uindex] * diff) * w_col[act_par_indexes] * r[uindex])
                        if np.float64(val) == 0:
                            logger.warning('\th = 0')

                if val:
                    values.append(val)
                    rows.append(mindex)
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

        for mindex, sequence in sequences.items():
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
                rows.append(mindex)
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
                vindex = user_map[str(v)]
                active_parents = sequence.get_active_parents(v, graph)
                if not active_parents:
                    continue
                act_par_indexes = [user_map[str(id)] for id in active_parents]
                act_par_times = np.matrix([[sequence.user_times[pid] for pid in active_parents]])
                user_time = np.repeat(np.matrix([sequence.user_times[v]]), len(active_parents))
                diff = user_time - act_par_times
                diff[diff == 0] = 1.0 / (24 * 60)  # 1 minute
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


def calc_r(sequences, graph, phi_h, psi, user_ids, cascade_map, user_map, m_set1, m_set2):
    try:
        r_values = []

        logger.info('\tcalculating values ...')
        i = 0
        for v in user_ids:
            v_i = user_map[str(v)]

            phi_sum = 0
            phi_time_sum = 0
            psi_time_sum = 0
            for m in set(m_set1[v]) | set(m_set2[v]):
                m_i = cascade_map[str(m)]
                cascade = sequences[m]
                active_parents = cascade.get_active_parents(v, graph)
                if not active_parents:
                    continue
                act_par_indexes = [user_map[str(id)] for id in active_parents]
                act_par_times = np.matrix([[cascade.user_times[pid] for pid in active_parents]])

                if m in m_set1[v]:
                    phi_h_col = phi_h[m_i][:, v_i].todense()
                    phi_sum += float(phi_h_col[act_par_indexes].sum())
                    user_time = np.repeat(np.matrix([cascade.user_times[v]]), len(active_parents))
                    diff = user_time - act_par_times
                    diff[diff == 0] = 1.0 / (24 * 60)  # 1 minute
                    phi_time_sum += float(diff * phi_h_col[act_par_indexes])

                if m in m_set2[v]:
                    psi_col = psi[m_i][:, v_i]
                    max_time = np.repeat(np.matrix([cascade.max_t]), len(active_parents))
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
            if len(user_ids) >= 10 and i % (len(user_ids) // 10) == 0:
                logger.debug('\t%d%% done', i * 100 // len(user_ids))

        return r_values
    except:
        logger.error(traceback.format_exc())
        raise


class Saito(AsLT):
    def __init__(self, project):
        self.project = project
        self.sample_count = 2500
        self.w_param_name = 'w-saito'
        self.r_param_name = 'r-saito'

        try:
            super(Saito, self).__init__(project)
        except FileNotFoundError:
            pass

    def calc_parameters(self, iterations=10):
        # Load dataset.
        logger.info('extracting data ...')
        train_set, _, _ = self.project.load_sets()
        graph, sequences = self.project.load_or_extract_graph_seq()

        # Create maps from users and cascades db id's to their matrix id's.
        logger.info('creating user and cascade id maps ...')
        db = DBManager(self.project.db).db
        user_ids = [u['_id'] for u in db.users.find({}, ['_id']).sort('_id')]
        user_map = {str(user_ids[i]): i for i in range(len(user_ids))}
        cascade_map = {str(train_set[i]): i for i in range(len(train_set))}
        logger.info('train set size = %d', len(train_set))
        logger.info('user space size = %d', len(user_map))

        # Set initial values of w and r.
        try:
            w = self.project.load_param(self.w_param_name, ParamTypes.SPARSE)
            r = self.project.load_param(self.r_param_name, ParamTypes.ARRAY)
            logger.info('w and r loaded')
        except:
            logger.info('initializing parameters ...')
            w, r = self.set_initial_values(graph, user_ids, user_map)
            self.project.save_param(w, self.w_param_name, ParamTypes.SPARSE)
            self.project.save_param(r, self.r_param_name, ParamTypes.ARRAY)

        # Run EM algorithm.
        logger.info('running algorithm ...')
        for i in range(iterations):
            with Timer('iteration time'):
                logger.info('#%d' % (i + 1))
                try:
                    h = self.project.load_param('h', ParamTypes.SPARSE)
                    logger.info('h loaded')
                except:
                    logger.info('calculating h ...')
                    h = self.calc_h_mp(sequences, graph, w, r, train_set, cascade_map, user_map)
                    self.project.save_param(h, 'h', ParamTypes.SPARSE)

                try:
                    g = self.project.load_param('g', ParamTypes.SPARSE)
                    logger.info('g loaded')
                except:
                    logger.info('calculating g ...')
                    g = self.calc_g_mp(sequences, graph, w, r, train_set, cascade_map, user_map)
                    self.project.save_param(g, 'g', ParamTypes.SPARSE)

                try:
                    self.project.load_param('phi_h', ParamTypes.SPARSE_LIST)  # Just check if exists.
                except:
                    logger.info('calculating phi_h ...')
                    # if i == 0 or len(user_map) ** 2 * len(train_set) < 10 ** 9:
                    # args = (sequences, graph, w, r, h, train_set, cascade_map, user_map)
                    # logger.debug('size of arguments: %f G', sum(asizeof(arg) for arg in args) / 1024 ** 3)
                    phi_h = self.calc_phi_h_mp(sequences, graph, w, r, h, train_set, cascade_map, user_map)
                    # else:
                    #     with Timer('calc_phi_h'):
                    #         seq2 = {cascade_map[str(cid)]: sequences[cid] for cid in sequences}
                    #         phi_h = calc_phi_h(seq2, graph, w, r, h, user_map)
                    self.project.save_param(phi_h, 'phi_h', ParamTypes.SPARSE_LIST)
                    del phi_h
                del h

                try:
                    self.project.load_param('phi_g', ParamTypes.SPARSE_LIST)  # Just check if exists.
                except:
                    logger.info('calculating phi_g ...')
                    # if i == 0 or len(user_map) ** 2 * len(train_set) < 10 ** 9:
                    # args = (sequences, graph, w, g, train_set, cascade_map, user_map)
                    # logger.debug('size of arguments: %f G', sum(asizeof(arg) for arg in args) / 1024 ** 3)
                    phi_g = self.calc_phi_g_mp(sequences, graph, w, g, train_set, cascade_map, user_map)
                    # else:
                    #     with Timer('calc_phi_g'):
                    #         seq2 = {cascade_map[str(cid)]: sequences[cid] for cid in sequences}
                    #         phi_g = calc_phi_g(seq2, graph, w, g, user_map)
                    self.project.save_param(phi_g, 'phi_g', ParamTypes.SPARSE_LIST)
                    del phi_g

                try:
                    psi = self.project.load_param('psi', ParamTypes.SPARSE_LIST)
                    logger.info('psi loaded')
                except:
                    logger.info('calculating psi ...')
                    # args = (sequences, graph, w, r, g, train_set, cascade_map, user_map)
                    # logger.debug('size of arguments: %f G', sum(asizeof(arg) for arg in args) / 1024 ** 3)
                    psi = self.calc_psi_mp(sequences, graph, w, r, g, train_set, cascade_map, user_map)
                    self.project.save_param(psi, 'psi', ParamTypes.SPARSE_LIST)

                del g
                phi_h = self.project.load_param('phi_h', ParamTypes.SPARSE_LIST)
                logger.info('phi_h loaded')

                logger.info('estimating r ...')
                last_r = r
                multi_processed = len(user_ids) < 2 * 10 ** 7
                r = self.calc_r_mp(sequences, graph, phi_h, psi, user_ids, train_set, cascade_map, user_map,
                                   multi_processed)

                phi_g = self.project.load_param('phi_g', ParamTypes.SPARSE_LIST)
                logger.info('phi_g loaded')

                logger.info('estimating w ...')
                last_w = w
                w = self.calc_w(sequences, graph, phi_h, phi_g, psi, user_ids, user_map, train_set, cascade_map)

                del phi_h
                del phi_g
                del psi

                # Save r and w.
                self.project.save_param(r, self.r_param_name, ParamTypes.ARRAY)
                self.project.save_param(w, self.w_param_name, ParamTypes.SPARSE)

                # Delete all except w and r.
                self.project.delete_param('h', ParamTypes.SPARSE)
                self.project.delete_param('g', ParamTypes.SPARSE)
                self.project.delete_param('phi_h', ParamTypes.SPARSE_LIST)
                self.project.delete_param('phi_g', ParamTypes.SPARSE_LIST)
                self.project.delete_param('psi', ParamTypes.SPARSE_LIST)

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

    @time_measure()
    def set_initial_values(self, graph, user_ids, user_map):
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
            if i % (u_count / 10) == 0:
                logger.info('%d%% done' % (i * 100 / u_count))

        w = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float32)
        r = np.ones(u_count, np.float32)
        return w, r

    @time_measure()
    def calc_h_mp(self, data, graph, w, r, cascade_ids, cascade_map, user_map):
        u_count = len(user_map)
        c_count = len(cascade_ids)
        process_count = min(settings.PROCESS_COUNT, c_count)
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

        h = sparse.csc_matrix((values, [rows, cols]), shape=(c_count, u_count), dtype=np.float64)
        return h

    @time_measure()
    def calc_g_mp(self, data, graph, w, r, cascade_ids, cascade_map, user_map):
        u_count = len(user_map)
        c_count = len(cascade_ids)
        process_count = min(settings.PROCESS_COUNT, c_count)
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

        g = sparse.csc_matrix((values, [rows, cols]), shape=(c_count, u_count), dtype=np.float32)
        return g

    @time_measure()
    def calc_phi_h_mp(self, data, graph, w, r, h, cascade_ids, cascade_map, user_map):
        c_count = len(cascade_ids)
        process_count = min(settings.PROCESS_COUNT, c_count)
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
        phi_h = [None for _ in range(len(cascade_ids))]
        for i in range(len(results)):
            phi_h_subset = results[i].get()
            for cindex, mat in phi_h_subset.items():
                phi_h[cindex] = mat

        return phi_h

    @time_measure()
    def calc_phi_g_mp(self, data, graph, w, g, cascade_ids, cascade_map, user_map):
        c_count = len(cascade_ids)
        process_count = min(settings.PROCESS_COUNT, c_count)
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
        phi_g = [None for _ in range(len(cascade_ids))]
        for i in range(len(results)):
            phi_g_subset = results[i].get()
            for cindex, mat in phi_g_subset.items():
                phi_g[cindex] = mat

        return phi_g

    @time_measure()
    def calc_psi_mp(self, data, graph, w, r, g, cascade_ids, cascade_map, user_map):
        c_count = len(cascade_ids)
        process_count = min(settings.PROCESS_COUNT, c_count)
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
        psi = [None for _ in range(len(cascade_ids))]
        for i in range(len(results)):
            psi_subset = results[i].get()
            for cindex, mat in psi_subset.items():
                psi[cindex] = mat

        return psi

    @time_measure()
    def calc_r_mp(self, data, graph, phi_h, psi, user_ids, cascade_ids, cascade_map, user_map, multi_processed=True):
        u_count = len(user_ids)

        logger.info('\textracting sigma domains ...')
        m_set1 = {v: [] for v in user_ids}
        m_set2 = {v: [] for v in user_ids}
        for m in cascade_ids:
            for v in data[m].users:
                m_set1[v].append(m)
            for v in data[m].get_rond_set(graph):
                m_set2[v].append(m)

        if multi_processed:
            logger.info('\tcreating processes to calculate values ...')
            process_count = min(settings.PROCESS_COUNT, u_count)
            pool = Pool(processes=process_count)
            step = int(math.ceil(u_count / process_count))
            results = []
            for j in range(0, u_count, step):
                subset = user_ids[j: j + step]
                m_set1_subset = {v: m_set1.pop(v) for v in subset}
                m_set2_subset = {v: m_set2.pop(v) for v in subset}
                res = pool.apply_async(calc_r, (
                    data, graph, phi_h, psi, subset, cascade_map, user_map, m_set1_subset, m_set2_subset))
                results.append(res)

            del m_set1
            del m_set2
            pool.close()
            pool.join()

            # Collect results of the processes.
            r_values = []
            for res in results:
                r_values.extend(res.get())
        else:
            r_values = calc_r(data, graph, phi_h, psi, user_ids, cascade_map, user_map, m_set1, m_set2)

        # Convert to numpy array.
        r = np.ones(u_count, np.float32)
        for i in range(u_count):
            r[user_map[str(user_ids[i])]] = r_values[i]

        return r

    @time_measure()
    def calc_w(self, data, graph, phi_h, phi_g, psi, user_ids, user_map, cascade_ids, cascade_map):
        u_count = len(user_ids)

        logger.info('\textracting sigma domains ...')
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

        logger.info('\tcalculating values ...')
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

        logger.info('\tnormalizing w ...')
        w = normalize(w, axis=0, copy=False)
        w = sparse.csc_matrix((w.data, w.indices, w.indptr), shape=w.shape, dtype=np.float32)

        return w
