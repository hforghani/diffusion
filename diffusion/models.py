import abc
from datetime import timedelta
from networkx import DiGraph
import numpy as np

from cascade.models import ParamTypes
from settings import logger
from utils.time_utils import str_to_datetime, DT_FORMAT


class LT(abc.ABC):
    def __init__(self, project):
        self.project = project
        self.init_tree = None
        self.max_delay = 24
        self.probabilities = {}  # dictionary of node id's to probabilities of activation
        self.w = None
        self.r = None
        self.w_param_name = None  # Must be implemented in subclasses.
        self.r_param_name = None  # Must be implemented in subclasses.

    def fit(self, multi_processed=False, eco=False, **kwargs):
        data_loaded = False
        if eco:
            try:
                self.w = self.project.load_param(self.w_param_name, ParamTypes.SPARSE).toarray()
                data_loaded = True
                self.r = self.project.load_param(self.r_param_name, ParamTypes.ARRAY)  # optional
            except FileNotFoundError:
                pass

        if not data_loaded:
            train_set, _, _ = self.project.load_sets()
            self.calc_parameters(train_set, multi_processed, eco, **kwargs)

        return self

    # @profile
    def predict(self, initial_tree, graph: DiGraph, thresholds: list = None, max_step: int = None) -> dict:
        """
        Predict activation cascade in the future starting from initial nodes given. Return a dict containing
        a tree for each threshold given.
        :param initial_tree:    initial tree of activated nodes
        :param graph:           DiGraph extracted from the training set
        :param thresholds:       list of thresholds of activation probability
        :param max_step:       maximum step to which prediction is done
        :return:    dictionary of thresholds to trees
        """
        if self.w is None:
            raise Exception('No w parameters found. Train the model first.')

        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        trees = {thr: initial_tree.copy() for thr in thresholds}

        # Initialize values.
        max_depth = initial_tree.depth
        cur_step_nodes = sorted(initial_tree.nodes_at_depth(max_depth),
                                key=lambda n: n.datetime)  # Set the nodes with maximum depth as initial step.
        max_thr = max(thresholds)
        cur_step = [(node, max_thr) for node in cur_step_nodes]
        active_ids = set(initial_tree.node_ids())
        self.probabilities = {}

        user_ids = sorted(graph.nodes())
        users_map = {user_ids[i]: i for i in range(len(user_ids))}

        # Iterate on steps. For each step try to activate other nodes.
        step = 1
        while cur_step and (max_step is None or step <= max_step):
            logger.debug('step %d on %d users ...', step, len(cur_step))

            next_step = []

            # Iterate on current step nodes to check if a child will be activated.
            for node, max_predicted_thr in cur_step:
                u = node.user_id  # sender user id
                if u not in users_map:
                    continue
                u_i = users_map[u]
                w_u = self.w[u_i, :]
                if w_u.any():
                    logger.debugv('weights of user %s :\n' + '\n'.join(
                        ['{} : {}'.format(i, w_u[i]) for i in np.nonzero(w_u)[0]]), u)

                # Iterate on children of u
                for v_i in np.nonzero(w_u)[0]:
                    v = user_ids[v_i]  # receiver (child) user id
                    if v in active_ids:
                        logger.debugv('user %s is already activated', v)
                        continue
                    prob = self.probabilities[v] = self.probabilities.get(v, 0) + w_u[v_i]
                    logger.debugv('probability of user %s = %s', v, prob)
                    child_max_pred_thr = None

                    for thr in thresholds:
                        if thr <= prob and thr <= max_predicted_thr:
                            if self.r is not None:
                                # Get delay parameter.
                                delay_param = self.r[v_i]

                                # Set the delay to mean of exponential distribution with parameter delay_param.
                                delay = 1 / delay_param if delay_param > 0 else self.max_delay  # in months
                                if delay > self.max_delay:
                                    delay = self.max_delay
                                send_dt = str_to_datetime(node.datetime)
                                receive_dt = (send_dt + timedelta(days=30 * delay)).strftime(DT_FORMAT)
                            else:
                                receive_dt = None

                            # Add it to the tree.
                            child = trees[thr].add_child(u, v, act_time=receive_dt)
                            child_max_pred_thr = thr
                            logger.debugv('a reshare predicted: prob (%f) >= thresh (%f)', self.probabilities[v], thr)

                    if child_max_pred_thr is not None:
                        next_step.append((child, child_max_pred_thr))
                        active_ids.add(v)

            cur_step = next_step
            if self.r is not None:
                cur_step = sorted(cur_step, key=lambda n: n[0].datetime)

            step += 1

        return trees

    @abc.abstractmethod
    def calc_parameters(self, train_set, multi_processed, eco, **kwargs):
        pass


class IC(abc.ABC):
    def __init__(self, project):
        self.project = project
        self.init_tree = None
        self.max_delay = 24
        self.probabilities = {}  # dictionary of node id's to probabilities of activation
        self.k = None
        self.r = None
        self.k_param_name = None  # Must be implemented in subclasses.
        self.r_param_name = None  # Must be implemented in subclasses.

    def fit(self, multi_processed=False, eco=False, **kwargs):
        data_loaded = False
        if eco:
            try:
                self.k = self.project.load_param(self.k_param_name, ParamTypes.SPARSE).tocsr()
                data_loaded = True
                if self.r_param_name is not None:
                    self.r = self.project.load_param(self.r_param_name, ParamTypes.SPARSE).tocsr()  # optional
            except FileNotFoundError:
                pass

        if not data_loaded:
            train_set, _, _ = self.project.load_sets()
            self.calc_parameters(train_set, multi_processed, eco, **kwargs)

        return self

    def predict(self, initial_tree, graph: DiGraph, thresholds: list = None, max_step: int = None) -> dict:
        """
        Predict activation cascade in the future starting from initial nodes given. Return a dict containing
        a tree for each threshold given.
        :param initial_tree:    initial tree of activated nodes
        :param graph:           DiGraph extracted from the training set
        :param thresholds:       list of thresholds of activation probability
        :param max_step:       maximum step to which prediction is done
        :return:    dictionary of thresholds to trees
        """
        if self.k is None:
            raise Exception('No k parameters found. Train the model first.')

        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        trees = {thr: initial_tree.copy() for thr in thresholds}

        # Initialize values.
        max_depth = initial_tree.depth
        cur_step_nodes = sorted(initial_tree.nodes_at_depth(max_depth),
                                key=lambda n: n.datetime)  # Set the nodes with maximum depth as initial step.
        max_thr = max(thresholds)
        cur_step = [(node, max_thr) for node in cur_step_nodes]
        active_ids = set(initial_tree.node_ids())
        self.probabilities = {}

        user_ids = sorted(graph.nodes())
        users_map = {user_ids[i]: i for i in range(len(user_ids))}

        # Iterate on steps. For each step try to activate other nodes.
        step = 1
        while cur_step and (max_step is None or step <= max_step):
            logger.debug('step %d on %d users ...', step, len(cur_step))

            next_step = []

            # Iterate on current step nodes to check if a child will be activated.
            for node, max_predicted_thr in cur_step:
                u = node.user_id  # sender user id
                if u not in users_map:
                    continue
                u_i = users_map[u]
                k_u = np.squeeze(self.k[u_i, :].toarray())  # probabilities of the children of u

                if k_u.any():
                    logger.debugv('probabilities of user %s :\n' + '\n'.join(
                        ['{} : {}'.format(i, k_u[i]) for i in np.nonzero(k_u)[0]]), u)

                # Iterate on children of u
                for v_i in np.nonzero(k_u)[0]:
                    v = user_ids[v_i]  # receiver (child) user id
                    if v in active_ids:
                        logger.debugv('user %s is already activated', v)
                        continue

                    k_u_v = k_u[v_i]
                    if v not in self.probabilities:
                        prob = self.probabilities[v] = k_u_v
                    else:
                        prob = self.probabilities[v] = max(k_u_v, self.probabilities[v])

                    logger.debugv('prob of user %s to user %s = %f', u, v, prob)
                    child_max_pred_thr = None

                    for thr in thresholds:
                        if thr <= prob and thr <= max_predicted_thr:
                            if self.r is not None:
                                # Get delay parameter.
                                delay_param = self.r[u_i, v_i]

                                # Set the delay to mean of exponential distribution with parameter delay_param.
                                delay = 1 / delay_param if delay_param > 0 else self.max_delay  # in months
                                if delay > self.max_delay:
                                    delay = self.max_delay
                                send_dt = str_to_datetime(node.datetime)
                                receive_dt = (send_dt + timedelta(days=30 * delay)).strftime(DT_FORMAT)
                            else:
                                receive_dt = None

                            # Add it to the tree.
                            child = trees[thr].add_child(u, v, act_time=receive_dt)
                            child_max_pred_thr = thr
                            logger.debugv('a reshare predicted: prob (%f) >= thresh (%f)', self.probabilities[v], thr)

                    if child_max_pred_thr is not None:
                        next_step.append((child, child_max_pred_thr))
                        active_ids.add(v)

            cur_step = next_step
            if self.r is not None:
                cur_step = sorted(cur_step, key=lambda n: n[0].datetime)

            step += 1

        return trees

    @abc.abstractmethod
    def calc_parameters(self, train_set, multi_processed, eco, **kwargs):
        pass
