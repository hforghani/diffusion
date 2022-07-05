from functools import reduce

import numpy as np
from scipy import sparse

from diffusion.enum import Method
from diffusion.models import DiffusionModel
from seq_labeling.utils import arr_to_str
from settings import logger


def filter_none_values(seq):
    return list(filter(lambda x: x is not None, seq))


class UnifiedMRFModel(DiffusionModel):
    method = Method.UNI_MRF
    max_iterations = 20

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, c2=0.1, eta=.1, epsilon=10 ** -6):
        super().__init__(initial_depth, max_step, threshold)
        self.c2 = c2
        self.eta = eta
        self.epsilon = epsilon
        self.__lambdaa = {}
        self.__nodes_map = {}

    def fit(self, train_set, train_trees, project, multi_processed=False, eco=False, **kwargs):
        super().fit(train_set, train_trees, project, multi_processed, eco)

        node_ids = list(self.graph.nodes())
        self.__nodes_map = {node_ids[i]: i for i in range(len(node_ids))}
        nodes_num = len(node_ids)
        max_depth = max(tree.depth for tree in train_trees)
        logger.debug('nodes_num = %d', nodes_num)
        logger.debug('max_depth = %d', max_depth)

        self.__lambdaa = self.__initialize_lambdaa(self.graph)
        states = self.__extract_states(train_trees, self.__nodes_map, max_depth)
        potential_sums = self.__calc_potential_sum(states, node_ids, self.__nodes_map, self.graph)

        for i in range(self.max_iterations):
            logger.info('iteration %d', i + 1)
            dif_values = np.zeros(len(self.__lambdaa))
            j = 0
            for node_id, lambdaa in self.__lambdaa.items():
                dif = self.eta * self.__likelihood_grad(node_id, lambdaa, self.__nodes_map, self.graph, states,
                                                        potential_sums[node_id])
                self.__lambdaa[node_id] += dif
                dif_values[j] = np.linalg.norm(dif)
                j += 1
            # logger.debug('dif values = %s', arr_to_str(dif_values))
            logger.info('dif max = %f', np.max(np.abs(dif_values)))
            # logger.debug('dif sum = %f', np.sum(np.abs(dif_values)))
            if np.max(dif_values) < self.epsilon:
                logger.info('stop criterion met')
                break

    def predict_one_sample(self, initial_tree, threshold, graph, max_step=None):
        if not isinstance(threshold, list):
            threshold = [threshold]

        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        trees = {thr: initial_tree.copy() for thr in threshold}

        # Initialize values.
        max_depth = initial_tree.depth
        cur_step_nodes = initial_tree.nodes_at_depth(max_depth)  # Set the nodes with maximum depth as the initial step.
        cur_step = {node.user_id for node in cur_step_nodes}
        init_nodes = set(initial_tree.node_ids())
        active_ids = {thr: init_nodes.copy() for thr in threshold}
        step_num = 1

        # timers = [Timer(f'code {i}', level='debug', silent=True) for i in range(10)]

        # Predict the cascade tree.
        # At each iteration find newly activated nodes based on the probabilities and add them to the tree.
        while cur_step and (max_step is None or step_num <= max_step):
            logger.debug('predicting step %d ...', step_num)
            next_step = set()

            # Get the children whom at least one of their parents are in current step.
            children_sets = (set(graph.successors(node_id)) for node_id in cur_step if node_id in graph)
            all_children = list(reduce(lambda x, y: x | y, children_sets, set()))

            j = 0
            for child_id in all_children:

                logger.debug('testing diffusion to %s ...', child_id)
                parents = list(graph.predecessors(child_id))
                activated = False
                last_pred = None

                for thr in threshold:
                    thr_active_ids = active_ids[thr]
                    parent_states = np.array([p in thr_active_ids for p in parents])
                    if child_id not in thr_active_ids:
                        logger.debug('threshold %f ...', thr)
                        new_active_ids = cur_step & thr_active_ids
                        if new_active_ids & set(parents):
                            parent_id, pred = self.__predict(thr, child_id, trees[thr], parent_states, new_active_ids,
                                                           parents, last_pred)
                            last_pred = pred
                            if parent_id:
                                node = trees[thr].add_node(child_id, parent_id=parent_id)
                                node.probability = pred.prob  # TODO: When running on large dataset remove this line.
                                thr_active_ids.add(child_id)
                                activated = True
                    else:
                        logger.debug('user %s is already activated', child_id)

                if activated:
                    next_step.add(child_id)

                j += 1
                if j % 100 == 0:
                    logger.debug('%d / %d of children iterated', j, len(all_children))
                    # for timer in timers:
                    #     if timer.sum:
                    #         timer.report_sum()

            cur_step = next_step
            step_num += 1

        return trees if len(trees) > 1 else next(iter(trees.values()))

    def __initialize_lambdaa(self, graph):
        return {node_id: np.ones(graph.in_degree(node_id)) for node_id in graph if graph.in_degree(node_id) > 0}

    def __likelihood_grad(self, node_id, lambdaa, nodes_map, graph, states, potential_sum):
        cascades_num = len(states)
        max_depth = states[0].shape[1]
        i = nodes_map[node_id]
        pred_indexes = [nodes_map[j] for j in graph.predecessors(node_id)]
        sum_over_t = np.zeros((cascades_num, len(pred_indexes)))
        # logger.debug('i = %d', i)
        # logger.debug('node_id = %s', node_id)
        # logger.debug('pred_indexes = %s', pred_indexes)
        for m in range(cascades_num):
            # logger.debug('m = %d', m)
            f_products = np.zeros((max_depth, len(pred_indexes)))
            for t in range(1, max_depth):
                # logger.debug('t = %d', t)
                s_m_i_t = states[m][i, t]
                pred_states = self.__state_array(states, m, pred_indexes, t - 1)
                # logger.debug('s_m_i_t = %s', s_m_i_t)
                # logger.debug('pred_states = %s', arr_to_str(pred_states))
                f = self.__potential(s_m_i_t, pred_states).astype(float)
                f_not = self.__potential(1 - s_m_i_t, pred_states).astype(float)
                product = f / (1 + np.exp(lambdaa.dot(f_not - f)))
                # logger.debug('f = %s', arr_to_str(f))
                # logger.debug('f_not = %s', arr_to_str(f_not))
                # logger.debug('product = %s', arr_to_str(product))
                f_products[t, :] = product
            sum_over_t[m, :] = np.sum(f_products, axis=0)
        sigma_grad_z = np.sum(sum_over_t, axis=0)
        # c2 = self.__c2 * len(pred_indexes)
        c2 = self.c2
        grad = potential_sum - sigma_grad_z - lambdaa / c2
        # logger.debug('potential_sum = %s', arr_to_str(potential_sum))
        # logger.debug('sigma_grad_z = %s', arr_to_str(sigma_grad_z))
        # logger.debug('lambdaa = %s', arr_to_str(lambdaa))
        # logger.debug('grad = %s', arr_to_str(grad))
        return grad

    def __extract_states(self, train_trees, nodes_map, max_depth):
        states = []
        cascades_num = len(train_trees)
        for m in range(cascades_num):
            tree = train_trees[m]
            states_m = sparse.lil_matrix((len(nodes_map), max_depth), dtype=bool)

            # Extract the "states" matrix which states[i,t] = 1 iff the node i is active at time t.
            active_nodes = tree.roots
            t = 0
            while active_nodes:
                indexes = filter_none_values(nodes_map.get(node.user_id) for node in active_nodes)
                states_m[indexes, t:] = 1
                active_nodes = reduce(lambda x, y: x + y, [node.children for node in active_nodes], [])
                t += 1

            states.append(states_m)
            # logger.debug('states_m = \n%s', two_d_arr_to_str(states_m.toarray()))

        return states

    def __calc_potential_sum(self, states, node_ids, nodes_map, graph):
        potential_sums = {}
        cascades_num = len(states)
        nodes_num, max_depth = states[0].shape
        for node_id in node_ids:
            i = nodes_map[node_id]
            pred_indexes = filter_none_values(nodes_map.get(j) for j in graph.predecessors(node_id))
            # logger.debug('node in-degree = %d', graph.in_degree(node_id))
            # logger.debug('pred_indexes = %s', pred_indexes)
            if pred_indexes:
                pot_sum_i = np.zeros(len(pred_indexes))
                for m in range(cascades_num):
                    # f_values = np.zeros((max_depth - 1, len(pred_indexes)), dtype=bool)
                    # for t in range(1, max_depth):
                    #     logger.debug('S_i^t = %s', states[m][i, t])
                    #     logger.debug('S_{Pa(i)}^{t-1} = %s',
                    #                  arr_to_str(self.__state_array(states, m, pred_indexes, t - 1)))
                    #     f = self.__potential(states[m][i, t], self.__state_array(states, m, pred_indexes, t - 1))
                    #     logger.debug('f.shape = %s', f.shape)
                    #     logger.debug('f(...) = %s', arr_to_str(f))
                    #     f_values[t - 1, :] = f
                    # pot_sum_i += np.sum(f_values, axis=0)
                    pot_sum_i += np.add.reduce(
                        [self.__potential(states[m][i, t], self.__state_array(states, m, pred_indexes, t - 1)) for t in
                         range(1, max_depth)])
                potential_sums[node_id] = pot_sum_i
        return potential_sums

    def __state_array(self, states: dict, m: int, node_indexes: list, t: int) -> np.ndarray:
        if len(node_indexes) == 1:
            return np.array([states[m][node_indexes[0], t]])
        else:
            return states[m][node_indexes, t].toarray().squeeze()

    def __potential(self, s: bool, s_pred: np.ndarray):
        return s_pred.copy() if s else np.zeros(s_pred.size, dtype=bool)
        # return s_pred.copy() if s else np.logical_not(s_pred)

    def __predict(self, thr, node_id, tree, parent_states, new_active_ids, parents, last_pred):
        if last_pred is not None and np.array_equal(parent_states, last_pred.parent_states):
            prob = last_pred.prob
        else:
            prob = self.__act_prob(parent_states, self.__lambdaa[node_id])
            logger.debug('prob = %f', prob)
        pred = Prediction(parent_states, prob)

        if prob >= thr:
            node_id = self.__predict_parent_id(node_id, new_active_ids, tree, parents)
            if node_id:
                logger.debug('a diffusion predicted from %s with prob %f >= %f', node_id, prob, thr)
            return node_id, pred

        return None, pred

    def __act_prob(self, pred_states, lambdaa):
        f = self.__potential(True, pred_states).astype(float)
        f_not = self.__potential(False, pred_states).astype(float)
        # logger.debug('pred_states = %s', arr_to_str(pred_states))
        # logger.debug('lambdaa = %s', arr_to_str(lambdaa))
        # logger.debug('f = %s', arr_to_str(f))
        # logger.debug('f_not = %s', arr_to_str(f_not))
        return 1 / (1 + np.exp(lambdaa.dot(f_not - f)))

    def __predict_parent_id(self, node_id, new_active_ids, tree, parents):
        """
        Set the parent with the maximum value of Lambda which is also activated at the current step as the predicted
        parent of this child.
        :param node_id: child node id to be activated
        :param new_active_ids: the node ids activated at the current step
        :param tree: current predicted tree
        :param parents: parent node ids
        :return: the predicted parent node id
        """
        # logger.debug('parents = %s', parents)
        new_active_parents = set(parents) & set(new_active_ids)
        # logger.debug('new_active_parents = %s', new_active_parents)
        new_active_par_indicator = [p in new_active_parents for p in parents]
        # logger.debug('new_active_par_indicator = %s', new_active_par_indicator)
        lambdaa_new_act = self.__lambdaa[node_id][new_active_par_indicator]
        # logger.debug('lambdaa_new_act = %s', arr_to_str(lambdaa_new_act))
        max_lambda_ind = np.argmax(lambdaa_new_act)
        # logger.debug('max_lambda_ind = %s', max_lambda_ind)
        predicted_node_id = [p for p in parents if p in new_active_parents][max_lambda_ind]
        # logger.debug('predicted_node_id = %s', predicted_node_id)
        if tree.get_node(predicted_node_id):
            return predicted_node_id
        else:
            logger.warning('parent node %s does not exist', predicted_node_id)


class Prediction:
    def __init__(self, parent_states, prob, state=True):
        self.parent_states = parent_states
        self.prob = prob
        self.state = state
