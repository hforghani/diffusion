import math

import numpy as np
from scipy import sparse

from cascade.models import IC, ParamTypes
from settings import logger
from utils.time_utils import time_measure


class DAIC(IC):
    def __init__(self, project):
        self.project = project
        self.k_param_name = 'k-daic'
        self.r_param_name = 'r-daic'

        try:
            super(DAIC, self).__init__(project)
        except FileNotFoundError:
            pass

    def calc_parameters(self, train_set, multi_processed, eco, **kwargs):
        iterations = kwargs.get('iterations', 10)
        if iterations is None:
            iterations = 10

        graph, sequences = self.project.load_or_extract_graph_seq()
        user_ids = sorted(graph.nodes())
        u_count = len(user_ids)
        user_map = {user_ids[i]: i for i in range(u_count)}
        cascade_map = {train_set[i]: i for i in range(len(train_set))}
        logger.info('train set size = %d', len(train_set))
        logger.info('user space size = %d', len(user_map))

        theta = self.__initialize(graph, user_ids, user_map)

        for it in range(iterations):
            logger.info('#%d', it + 1)

            p = self.__calc_p(theta, train_set, user_ids, user_map, cascade_map, graph, sequences)

            last_theta = theta.copy()
            theta = self.__calc_theta(p, theta, user_ids, user_map, cascade_map, graph, sequences)

            theta_dif = theta - last_theta
            theta_dif = np.sqrt(np.multiply(theta_dif, theta_dif).sum())
            del last_theta
            logger.info('theta dif = %f', theta_dif)

            if eco:
                self.project.save_param(sparse.csr_matrix(theta), self.k_param_name, ParamTypes.SPARSE)

        self.k = theta

    def __initialize(self, graph, user_ids, user_map):
        # Initialize probabilities.
        u_count = len(user_ids)
        theta = np.zeros((u_count, u_count))
        for i in range(u_count):
            child_indexes = [user_map[uid] for uid in graph.successors(user_ids[i])]
            if child_indexes:
                theta[i, child_indexes] = 1 / len(child_indexes)
        return theta

    @time_measure('debug')
    def __calc_p(self, theta, cascade_ids, user_ids, user_map, cascade_map, graph, sequences):
        c_count = len(cascade_ids)
        u_count = len(user_ids)
        p = np.zeros((c_count, u_count))

        for cid in cascade_ids:
            logger.debug('calculating p: cascade %s ...', cid)
            cindex = cascade_map[cid]
            seq = sequences[cid]

            for v in seq.users:
                # logger.debug('calculating p: user %s ...', v)
                if v not in user_map:
                    continue
                vindex = user_map[v]
                prev_par_indexes = [user_map[p] for p in seq.get_active_parents(v, graph)]
                if prev_par_indexes:
                    p[cindex, vindex] = 1 - np.prod(1 - theta[prev_par_indexes, vindex])
                    # logger.debug('prev_par_indexes = %s', prev_par_indexes)
                    # logger.debug('theta[prev_par_indexes, vindex] = %s', theta[prev_par_indexes, vindex])
                    # logger.debug('p[cindex, vindex] = %f', p[cindex, vindex])

        return p

    @time_measure('debug')
    def __calc_theta(self, p, theta, user_ids, user_map, cascade_map, graph, sequences):
        u_count = len(user_ids)
        new_theta = np.zeros((u_count, u_count))
        lambdaa = 10

        for u in user_ids:
            logger.debug('calculating theta: user %s ...', u)
            u_index = user_map[u]

            for v in graph.successors(u):
                v_index = user_map[v]
                beta, positives = self.__calc_beta(u, v, lambdaa, sequences)
                gamma = self.__calc_gamma(v_index, p, theta[u_index, v_index], cascade_map, positives)
                delta = beta ** 2 - 4 * lambdaa * gamma
                new_theta[u_index, v_index] = (beta - math.sqrt(delta)) / (2 * lambdaa)

        return new_theta

    # @time_measure('debug')
    def __calc_beta(self, u, v, lambdaa, sequences):
        positives = []
        negatives = []
        for cid, seq in sequences.items():
            ui = seq.users.index(u) if u in seq.users else None
            if ui:
                vi = seq.users.index(v) if v in seq.users else None
            else:
                vi = None
            if ui is not None:
                if vi is None:
                    negatives.append(cid)
                elif seq.times[ui] < seq.times[vi]:
                    positives.append(cid)
        beta = len(positives) + len(negatives) + lambdaa
        return beta, positives

    # @time_measure('debug')
    def __calc_gamma(self, user_index, p, theta_u_v, cascade_map, positives):
        gamma = 0
        for cid in positives:
            gamma += 1 / p[cascade_map[cid], user_index]
        gamma *= theta_u_v
        return gamma
