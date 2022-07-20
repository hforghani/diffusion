import math
import pprint
import typing
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, repeat
from random import random

from matplotlib import pyplot as plt
import networkx
import numpy as np
from networkx import Graph
from scipy import sparse

import settings
from diffusion.enum import Method
from diffusion.models import DiffusionModel
from seq_labeling.utils import arr_to_str, two_d_arr_to_str
from settings import logger
from utils.time_utils import Timer

timers = [Timer(f'code {i}', level='debug', silent=True) for i in range(10)]


def filter_none_values(seq):
    return list(filter(lambda x: x is not None, seq))


def report_sum():
    for timer in timers:
        if timer.sum:
            timer.report_sum()


class UnifiedMRFModel(DiffusionModel):
    method = Method.UNI_MRF
    max_iterations = 20

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, c2=10, eta=.01, epsilon=10 ** -6, **kwargs):
        super().__init__(initial_depth, max_step, threshold, **kwargs)
        self.c2 = c2
        self.eta = eta
        self.epsilon = epsilon
        self.gibbs_samples_num = 10 ** 5
        self.__nodes_map = {}
        self.__lambda = []
        # self.__predecessors = []
        # self.__psi = {}

    def fit(self, train_set, train_trees, project, multi_processed=False, eco=False, iterations=None, **kwargs):
        super().fit(train_set, train_trees, project, multi_processed, eco)

        node_ids = list(self.graph.nodes())
        nodes_map = {node_ids[i]: i for i in range(len(node_ids))}
        nodes_num = len(node_ids)
        logger.debug('nodes_num = %d', nodes_num)

        # self.__predecessors = [list(self.graph.predecessors(node_ids[i])) for i in range(nodes_num)]
        pred_indexes = [[nodes_map[n] for n in self.graph.predecessors(node_ids[i])] for i in range(nodes_num)]
        child_indexes = [{nodes_map[n] for n in self.graph.successors(node_ids[i])} for i in range(nodes_num)]
        lambdaa = self.__initialize_lambdaa(pred_indexes)
        states = self.__extract_states(train_trees, nodes_map)
        data_size = len(states)
        logger.debug('data_size = %d', data_size)
        feature_sums = self.__feature_sums(states, pred_indexes, nodes_num)
        # junction_tree = self.__create_junction_tree(node_ids, nodes_map, self.__train_max_depth)
        not_none_indexes = [i for i in range(nodes_num) if lambdaa[i] is not None]
        max_iterations = iterations if iterations is not None else self.max_iterations

        for it in range(max_iterations):
            logger.info('iteration %d', it + 1)
            # logger.debug('running Hugin ...')
            # psi = self.__run_hugin(junction_tree, node_ids, self.__train_max_depth, lambdaa, pred_indexes)

            if multi_processed:
                gradients = self.__likelihood_grad_mp(not_none_indexes, child_indexes, data_size, feature_sums, lambdaa,
                                                      pred_indexes, self.c2, self.gibbs_samples_num)
            else:
                gradients = self.likelihood_gradients(not_none_indexes, pred_indexes, child_indexes, data_size,
                                                      feature_sums, lambdaa, self.c2, self.gibbs_samples_num)

            new_lambda = [None] * nodes_num
            dif_values = np.zeros(nodes_num)
            logger.debug('new lambda =')
            for j in range(len(not_none_indexes)):
                i = not_none_indexes[j]
                dif = self.eta * gradients[j]
                new_lambda_i = lambdaa[i] + dif
                new_lambda[i] = new_lambda_i
                # logger.debug('lambda = %s', arr_to_str(lambdaa[i]))
                # logger.debug('dif = %s', arr_to_str(dif))
                logger.debug('%d -> %s', i, arr_to_str(new_lambda_i))
                dif_values[i] = np.linalg.norm(dif)

            lambdaa = new_lambda
            # logger.debug('all lambda = \n%s', pprint.pformat(lambdaa))
            # logger.debug('dif values = %s', arr_to_str(dif_values))
            logger.info('dif max = %f', np.max(np.abs(dif_values)))
            logger.debug('dif sum = %f', np.sum(np.abs(dif_values)))
            if np.max(dif_values) < self.epsilon:
                logger.info('stop criterion met')
                break

        # self.__psi = self.__run_hugin(junction_tree, node_ids, self.__train_max_depth, lambdaa, pred_indexes-)
        self.__nodes_map = nodes_map
        self.__lambda = lambdaa
        return self

    def __likelihood_grad_mp(self, not_none_indexes, child_indexes, data_size, feature_sums, lambdaa, pred_indexes, c2,
                             gibbs_samples_num):
        bunch_size = min(100, math.ceil(len(not_none_indexes / settings.TRAIN_WORKERS)))
        indexes_bunches = [not_none_indexes[i:i + bunch_size] for i in range(0, len(not_none_indexes), bunch_size)]
        with ProcessPoolExecutor(max_workers=settings.TRAIN_WORKERS) as executor:
            grad_bunches = executor.map(likelihood_grad, indexes_bunches, repeat(pred_indexes), repeat(child_indexes),
                                        repeat(data_size), repeat(feature_sums), repeat(lambdaa), repeat(c2),
                                        repeat(gibbs_samples_num))
        return list(chain(*grad_bunches))

    def predict_one_sample(self, initial_tree, threshold, graph, max_step=None):
        if not isinstance(threshold, list):
            threshold = [threshold]

        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        trees = {thr: initial_tree.copy() for thr in threshold}

        # Initialize values.
        init_depth = initial_tree.depth
        cur_step_nodes = initial_tree.nodes_at_depth(
            init_depth)  # Set the nodes with maximum depth as the initial step.
        cur_step = {node.user_id for node in cur_step_nodes}
        init_nodes = set(initial_tree.node_ids())
        active_ids = {thr: init_nodes.copy() for thr in threshold}
        step_num = 1

        # timers = [Timer(f'code {i}', level='debug', silent=True) for i in range(10)]

        # Predict the cascade tree.
        # At each iteration find newly activated nodes based on the probabilities and add them to the tree.
        while cur_step and (max_step is None or step_num <= max_step):
            logger.debug('predicting step %d ...', step_num)
            # cur_depth = init_depth + step_num
            # if cur_depth == self.__train_max_depth:
            #     break
            cur_depth = min(init_depth + step_num, self.__train_max_depth - 1)
            next_step = set()

            # Get the children whom at least one of their parents are in current step.
            children_sets = (set(graph.successors(node_id)) for node_id in cur_step if node_id in graph)
            all_children = list(set().union(*children_sets))

            j = 0
            for child_id in all_children:

                logger.debug('testing diffusion to %s ...', child_id)
                parents = self.__predecessors[child_id]
                activated = False
                last_pred = None

                for thr in threshold:
                    thr_active_ids = active_ids[thr]
                    parent_states = np.array([p in thr_active_ids for p in parents])
                    if child_id not in thr_active_ids:
                        logger.debug('threshold %f ...', thr)
                        new_active_ids = cur_step & thr_active_ids
                        if new_active_ids & set(parents):
                            parent_id, pred = self.__predict(thr, child_id, cur_depth, trees[thr],
                                                             parent_states, new_active_ids, parents, last_pred)
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

    def __initialize_lambdaa(self, pred_indexes):
        nodes_num = len(pred_indexes)
        pred_nums = [len(pred_list) for pred_list in pred_indexes]
        return [np.ones(pred_nums[i] + 1) if pred_nums[i] else None for i in range(nodes_num)]

    @classmethod
    def __likelihood_grad(cls, node_index, pred_indexes, child_indexes, data_size, feature_sum, lambdaa, c2,
                          gibbs_samples_num):
        logger.debug('calculating grad_ln_z for node index %d ...', node_index)
        grad_ln_z = cls.__grad_ln_z(node_index, pred_indexes, child_indexes, lambdaa, gibbs_samples_num)
        grad = feature_sum - data_size * grad_ln_z - lambdaa[node_index] / c2
        logger.debug('potential_sum = %s', arr_to_str(feature_sum))
        logger.debug('M * grad_ln_z = %s', arr_to_str(data_size * grad_ln_z))
        logger.debug('lambda / c2 = %s', arr_to_str(lambdaa[node_index] / c2))
        logger.debug('grad = %s', arr_to_str(grad))
        return grad

    def __extract_states(self, train_trees, nodes_map):
        """
        Extract the state matrices of the training data.
        :param train_trees: list of training set trees
        :param nodes_map: dict of node ids to node indexes
        :return: list of N * T sparse matrices each corresponding to a cascade. N is number of nodes and T is the
        maximum depth. states_k[i,t] = 1 iff the node i is active at time t.
        """
        nodes_num = len(nodes_map)
        states = []
        for tree in train_trees:
            cur_nodes = tree.roots
            last_states = None
            while cur_nodes:
                indexes = filter_none_values(nodes_map.get(node.user_id) for node in cur_nodes)
                if last_states is None:
                    cur_states = np.zeros((nodes_num, 2), dtype=bool)
                else:
                    cur_states = np.tile(last_states[:, 1].reshape((nodes_num, 1)), (1, 2))
                cur_states[indexes, 1] = True
                if last_states is not None:
                    states.append(cur_states)
                    # logger.debug('cur_states = \n%s', two_d_arr_to_str(cur_states))
                cur_nodes = list(chain(*[node.children for node in cur_nodes]))
                last_states = cur_states

        return states

    def __feature_sums(self, states, pred_indexes, nodes_num):
        """
        Calculate the feature sum for each node id.
        :param states: list of training states matrices
        :param node_ids: list of training node ids
        :param nodes_map: dict of node ids to their indexes
        :return: dict of node ids to their feature sum terms appearing in the likelihood gradient formula.
        for node id "i", the potential sum is:
        \sum_{m=1}^M \sum_{t=1}^T f \left( {S_i^t}^{(m)}, {S_{Pa(i)}^{t-1}}^{(m)}\right)
        """
        logger.debug('calculating feature sums ...')
        feat_sums = []
        data_size = len(states)
        for i in range(nodes_num):
            pred_indexes_i = pred_indexes[i]
            pred_num = len(pred_indexes[i])
            # logger.debug('node in-degree = %d', pred_num)
            # logger.debug('pred_indexes[i] = %s', pred_indexes[i])
            if pred_num:
                feat_sum_i = np.add.reduce([
                    self.__feature(states[m][([i, i] + pred_indexes_i, [1] + [0] * (pred_num + 1))], pred_num)
                    for m in range(data_size)
                ])
                # for m in range(data_size):
                #     logger.debug('value of click i:\n%s',
                #                  arr_to_str(states[m][([i, i] + pred_indexes_i, [1] + [0] * (pred_num + 1))]))
                # logger.debug('feat_sum_i = %s', arr_to_str(feat_sum_i))
            else:
                feat_sum_i = 0
            feat_sums.append(feat_sum_i)
        # logger.debug('feat_sums = \n%s', pprint.pformat(feat_sums))
        return feat_sums

    def __predict(self, thr, node_id, cur_depth, tree, parent_states, new_active_ids, parents, last_pred):
        if last_pred is not None and np.array_equal(parent_states, last_pred.parent_states):
            prob = last_pred.prob
        else:
            node_index = self.__nodes_map[node_id]
            prob = self.__act_prob(node_index, cur_depth, parent_states)
            logger.debug('prob = %f', prob)
        pred = Prediction(parent_states, prob)

        if prob >= thr:
            node_id = self.__predict_parent_id(node_id, new_active_ids, tree, parents)
            if node_id:
                logger.debug('a diffusion predicted from %s with prob %f >= %f', node_id, prob, thr)
            return node_id, pred

        return None, pred

    @staticmethod
    def __feature(value: typing.Union[int, np.ndarray], pred_num: int):
        """
        Get the feature vector f(S_i_t, S_Pa(i)_(t-1)).
        :return: numpy array of the feature
        """
        if isinstance(value, int):
            # logger.debug('value = %d, pred_num = %d', value, pred_num)
            if value >= 2 ** (pred_num + 1):
                f = np.array([d == '1' for d in f'{bin(value % (2 ** (pred_num + 1)))[2:]:0>{pred_num + 1}}'])
            else:
                f = np.zeros(pred_num + 1, dtype=bool)
            # logger.debug('feature = %s', arr_to_str(f))
        elif isinstance(value, np.ndarray):
            f = value[1:].astype(float) if value[0] else np.zeros(pred_num + 1, dtype=bool)
        else:
            raise TypeError('value must be an integer or numpy.ndarray')
        return f

    def __act_prob(self, node_index, cur_depth, pred_states):
        pred_num = len(pred_states)
        value_for_active = np.zeros(pred_num + 2, dtype=bool)
        value_for_active[0:2] = [True, False]
        value_for_active[2:] = pred_states
        psi = self.__psi[(node_index, cur_depth)]
        return self.__joint_prob_i_t(value_for_active, pred_num, psi)

        # value_for_inactive = np.zeros(pred_num + 2, dtype=bool)
        # value_for_inactive[1] = False
        # value_for_inactive[2:] = pred_states
        # value_for_active = value_for_inactive.copy()
        # value_for_active[0] = True
        # f_active = self.__feature(value_for_active, pred_num).astype(float)
        # f_inactive = self.__feature(value_for_inactive, pred_num).astype(float)
        # # logger.debug('pred_states = %s', arr_to_str(pred_states))
        # # logger.debug('lambdaa = %s', arr_to_str(lambdaa))
        # # logger.debug('f_active = %s', arr_to_str(f_active))
        # # logger.debug('f_inactive = %s', arr_to_str(f_inactive))
        # return 1 / (1 + np.exp(lambdaa.dot(f_inactive - f_active)))

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
        new_active_par_indicator = [p in new_active_ids for p in parents]
        # logger.debug('new_active_par_indicator = %s', new_active_par_indicator)
        lambdaa_new_act = self.__lambda[self.__nodes_map[node_id]][1:][new_active_par_indicator]
        # logger.debug('lambdaa_new_act = %s', arr_to_str(lambdaa_new_act))
        max_lambda_ind = np.argmax(lambdaa_new_act)
        # logger.debug('max_lambda_ind = %s', max_lambda_ind)
        predicted_node_id = [p for p in parents if p in new_active_ids][max_lambda_ind]
        # logger.debug('predicted_node_id = %s', predicted_node_id)
        if tree.get_node(predicted_node_id):
            return predicted_node_id
        else:
            logger.warning('parent node %s does not exist', predicted_node_id)

    def __create_junction_tree(self, node_ids, nodes_map, max_depth):
        # Create click graph.
        nodes_num = len(nodes_map)
        logger.debug('creating click graph ...')
        click_graph = Graph()
        click_graph.add_edges_from(
            [((nodes_map[j], 1), (i, 2)) for i in range(nodes_num) for j in
             self.__predecessors[node_ids[i]] + [node_ids[i]]])
        mst = networkx.minimum_spanning_tree(click_graph)
        logger.debug('num of click graph nodes, edges = %d, %d', click_graph.number_of_nodes(),
                     click_graph.number_of_edges())
        logger.debug('num of MST nodes, edges = %d, %d', mst.number_of_nodes(), mst.number_of_edges())
        # logger.debug('MST edges: \n%s', pprint.pformat(sorted(list(mst.edges()))))
        # logger.debug('drawing MST subgraph ...')
        # subgraph = mst.subgraph([(i, t) for i in range(50) for t in [2, 3]])
        # pos = {n: (n[1] * 2 / max_depth - 1, 1 - n[0] * 2 / nodes_num) for n in subgraph.nodes()}
        # plt.figure(0, figsize=(200, 80), dpi=60)
        # networkx.draw_networkx(subgraph, pos)
        # plt.savefig('mst.jpg', format="JPG")

        # Create junction tree.
        logger.debug('adding the rest of junction tree edges ...')
        j_tree = mst.copy()
        j_tree.add_edges_from(
            [((i, t - 1), (i, t)) for i in range(nodes_num) for t in range(3, max_depth)]
        )
        logger.debug('num of junction tree nodes, edges = %d, %d', j_tree.number_of_nodes(), j_tree.number_of_edges())
        # logger.debug('junction tree edges: \n%s', pprint.pformat(sorted(list(j_tree.edges()))))
        # logger.debug('drawing junction subtree ...')
        # subgraph = j_tree.subgraph([(i, t) for i in range(50) for t in range(1, max_depth)])
        # pos = {n: (n[1] * 2 / max_depth - 1, 1 - n[0] * 2 / nodes_num) for n in subgraph.nodes()}
        # plt.figure(1, figsize=(200, 80), dpi=60)
        # networkx.draw_networkx(subgraph, pos)
        # plt.savefig('junction_tree.jpg', format="JPG")
        # logger.debug('done')
        return j_tree

    def __joint_prob_i_t(self, values_i_t: typing.Union[int, np.ndarray], pred_num, psi_i_t):
        """
        Compute the probability P(S_i^t, S_Pa(i)^(t-1)).
        :return:
        """
        # logger.debug('pred_num = %d', pred_num)
        # logger.debug('value = %d', values_i_t)

        # If the last state is active, the current step must be deterministically 1.
        if isinstance(values_i_t, int):
            s_i_t_minus_1 = (values_i_t >> pred_num) % 2
            if s_i_t_minus_1:  # S_i^{t - 1} = True
                return values_i_t > 2 ** (pred_num + 1)
        elif isinstance(values_i_t, np.ndarray):
            if values_i_t[1]:  # S_i^{t - 1} = True
                return 1 if values_i_t[0] else 0
        else:
            raise TypeError('values_i_t must be an integer or numpy.ndarray')

        if isinstance(values_i_t, np.ndarray):
            # logger.debug('values_i_t = %s', arr_to_str(values_i_t))
            values_i_t = sum(values_i_t[i] * 2 ** (pred_num + 1 - i) for i in range(pred_num + 2))
            # logger.debug('values_i_t converted to %d', values_i_t)

        prob = psi_i_t[values_i_t] / np.sum(psi_i_t)
        # logger.debug('psi_i_t = %s', arr_to_str(psi_i_t))
        # logger.debug('P(C_%d^%d = %d) = %f', i, t, values_i_t, prob)
        # if i % 10 == 0 and (t, values_i_t) == (2, 0):
        #     report_sum()
        return prob

    def __run_hugin(self, junction_tree, node_ids, max_depth, lambdaa, pred_indexes):
        logger.debug('initializing Hugin ...')
        phi, psi = self.__initialize_hugin(node_ids, max_depth, lambdaa, pred_indexes)
        root = (0, 1)
        bfs_tree = networkx.bfs_tree(junction_tree, root)
        logger.debug('belief propagation from leaves to root ...')
        self.__collect_evidence(root, bfs_tree, node_ids, phi, psi)
        logger.debug('belief propagation from root to leaves ...')
        self.__distribute_evidence(root, bfs_tree, node_ids, phi, psi)
        logger.debug('Hugin completed')
        return psi

    def __initialize_hugin(self, node_ids, max_depth, lambdaa, pred_indexes):
        nodes_num = len(node_ids)
        phi = {(i, t): np.ones(2) for i in range(nodes_num) for t in range(1, max_depth)}
        psi = {}
        for i in range(nodes_num):
            lambdaa_i = lambdaa[i]
            if lambdaa_i is None:
                lambdaa_i = np.ones(1)
            # logger.debug('i = %d', i)
            # logger.debug('size of lambdaa = %s', lambdaa_i.size)
            # logger.debug('lambdaa = %s', arr_to_str(lambdaa_i))
            pred_num = len(pred_indexes[i])
            # TODO: correct only with binary/zero feature function.
            psi_i_t = np.zeros(2 ** (2 + pred_num))
            half_size = int(psi_i_t.size / 2)
            # logger.debug('pred_num = %d', pred_num)
            # logger.debug('size of psi_i_t = %d', psi_i_t.size)
            for ind in range(1 + pred_num):
                step = 2 ** (pred_num - ind)
                # logger.debug('ind = %d -> step = %d', ind, step)
                for j in range(int(psi_i_t.size / step)):
                    if j % 2 == 1:
                        psi_i_t[half_size + j * step: half_size + (j + 1) * step] += lambdaa_i[
                            ind]  # value of X_ind = 1
            # logger.debug('psi_i_t before exp = %s', arr_to_str(psi_i_t))
            psi_i_t = np.exp(psi_i_t)
            # logger.debug('psi_i_t = %s', arr_to_str(psi_i_t))

            for t in range(1, max_depth):
                psi[(i, t)] = psi_i_t

        return phi, psi

    def __collect_evidence(self, click, bfs_tree, node_ids, phi, psi):
        for child in bfs_tree.successors(click):
            self.__collect_evidence(child, bfs_tree, node_ids, phi, psi)
            self.__update(child, click, node_ids, phi, psi)

    def __distribute_evidence(self, click, bfs_tree, node_ids, phi, psi):
        for child in bfs_tree.successors(click):
            self.__update(click, child, node_ids, phi, psi)
            self.__distribute_evidence(child, bfs_tree, node_ids, phi, psi)

    def __update(self, src, dst, node_ids, phi, psi):
        # logger.debug('updating from %s to %s', src, dst)
        i_src, t_src = src
        i_dst, t_dst = dst
        sep = src if t_src < t_dst else dst
        phi_sep_new = self.__marginalize(src, sep, psi[src], node_ids)
        psi_dst_new = self.__rescale(sep, dst, phi[sep], phi_sep_new, psi[dst], node_ids)
        psi[dst] = psi_dst_new
        phi[sep] = phi_sep_new

    def __marginalize(self, src, sep, psi_src, node_ids):
        i_src, t_src = src
        sep_ind = self.__sep_index_in_click(sep, src, node_ids)
        # logger.debug('src = %s', src)
        # logger.debug('sep = %s', sep)
        # logger.debug('sep_ind = %d', sep_ind)
        # logger.debug('psi_src = %s', arr_to_str(psi_src))
        phi_sep_new = np.zeros(2)
        # logger.debug('src in-degree = %d', len(self.__predecessors[node_ids[i_src]]))
        step = int(2 ** (len(self.__predecessors[i_src]) + 1 - sep_ind))
        # logger.debug('step = %d', step)
        for i in range(int(psi_src.size / step)):
            # logger.debug('%s to %s', i * step, (i + 1) * step)
            part_sum = psi_src[i * step: (i + 1) * step].sum()
            if i % 2 == 0:
                phi_sep_new[0] += part_sum  # value of X_sep = 0
            else:
                phi_sep_new[1] += part_sum  # value of X_sep = 1
        # logger.debug('phi_sep_new = %s', arr_to_str(phi_sep_new))
        return phi_sep_new

    def __rescale(self, sep, dst, phi_sep, phi_sep_new, psi_dst, node_ids):
        sep_ind = self.__sep_index_in_click(sep, dst, node_ids)
        scale_values = np.divide(phi_sep_new, phi_sep)
        scale_values[np.isnan(scale_values)] = 0
        i_dst, t_dst = dst
        scale = np.zeros(psi_dst.size)
        step = 2 ** (len(self.__predecessors[i_dst]) + 1 - sep_ind)
        # logger.debug('dst = %s', dst)
        # logger.debug('sep = %s', sep)
        # logger.debug('sep_ind = %d', sep_ind)
        # logger.debug('psi_dst = %s', arr_to_str(psi_dst))
        # logger.debug('phi_sep = %s', arr_to_str(phi_sep))
        # logger.debug('phi_sep_new = %s', arr_to_str(phi_sep_new))
        # logger.debug('src in-degree = %d', len(self.__predecessors[node_ids[i_dst]]))
        # logger.debug('step = %d', step)

        for i in range(int(psi_dst.size / step)):
            # logger.debug('%s to %s', i * step, (i + 1) * step)
            if i % 2 == 0:
                scale[i * step: (i + 1) * step] = scale_values[0]  # X_sep = 0
            else:
                scale[i * step: (i + 1) * step] = scale_values[1]  # X_sep = 1

        # logger.debug('scale = %s', arr_to_str(scale))
        psi_new = np.multiply(psi_dst, scale)
        # logger.debug('psi_new = %s', arr_to_str(psi_new))
        return psi_new

    def __sep_index_in_click(self, sep, src, node_ids):
        i_src, t_src = src
        i_sep, t_sep = sep
        if sep == src:
            return 0
        elif sep == (i_src, t_src - 1):
            return 1
        else:
            return 2 + self.__predecessors[node_ids[i_src]].index(node_ids[i_sep])

    @classmethod
    def __grad_ln_z(cls, node_index, pred_indexes, child_indexes, lambdaa, gibbs_samples_num):
        i = node_index
        # logger.debug('Gibbs sampling ...')
        with timers[0]:
            samples = cls.__gibbs_samples(gibbs_samples_num, pred_indexes, child_indexes, lambdaa)
        # logger.debug('Gibbs samples = \n%s \n...', pprint.pformat(samples[:10]))
        # logger.debug('calculating joint probabilities ...')
        with timers[1]:
            pot_sigma = cls.__gibbs_samples_pot_sigma(samples, pred_indexes, child_indexes, lambdaa)
        # logger.debug('pot_sigma = %s', pot_sigma)
        with timers[2]:
            joint_probs = cls.__normalize_pot_sigma(pot_sigma)
        # logger.debug('normalized joint_probs = %s', joint_probs)
        pred_indexes_i = pred_indexes[i]
        # logger.debug('creating sample features ...')
        with timers[3]:
            # features = cls.__gibbs_samples_feat_i(samples, i, pred_indexes_i)
            pred_num = len(pred_indexes_i)
            features = np.array(
                [cls.__feature(value[([i, i] + pred_indexes_i, [1] + [0] * (pred_num + 1))], pred_num) for value in
                 samples])
        # logger.debug('features = \n%s', features)
        with timers[4]:
            grad_ln_z = joint_probs.dot(features)
        # logger.debug('grad_ln_z = \n%s', arr_to_str(grad_ln_z))
        # report_sum()
        return grad_ln_z

    @classmethod
    def __gibbs_samples(cls, samples_num, pred_indexes, child_indexes, lambdaa):
        nodes_num = len(pred_indexes)
        # sample = np.ones((nodes_num, 2), dtype=bool)
        sample = np.zeros((nodes_num, 2), dtype=bool)
        # logger.debug('sample.T = \n%s', two_d_arr_to_str(sample.T))
        samples = [sample]
        variables = [(i, t) for i in range(nodes_num) for t in range(2)]
        count = 1

        while count < samples_num:
            for var in variables:
                # logger.debug('var = %s', var)
                i, t = var
                if (t == 1 and sample[i, 0]) or not pred_indexes[i]:
                    sample = sample.copy()
                else:
                    sigma_dif = cls.__sigma_dif(sample, var, pred_indexes, child_indexes, lambdaa)
                    prob1 = 1 / (1 + np.exp(-sigma_dif))
                    # logger.debug('sigma_dif = %f', sigma_dif)
                    # logger.debug('prob1 = %f', prob1)
                    sample = sample.copy()
                    if random() < prob1:
                        sample[var] = True
                    else:
                        sample[var] = False
                # logger.debug('sample.T = \n%s', two_d_arr_to_str(sample.T))
                samples.append(sample)
                count += 1
                # if count % 10 ** 5 == 0:
                #     logger.debug('%d%% done', count / samples_num * 100)
                if count >= samples_num:
                    break

        return samples

    @staticmethod
    def __potentials_sigma(value, pred_indexes, lambdaa):
        # logger.debug('value.T = \n%s', two_d_arr_to_str(value.T))
        pot_sum = np.array([lambdaa[i][value[[i] + pred_indexes[i], 0]].sum()
                            for i in value[:, 1].nonzero()[0]
                            if lambdaa[i] is not None]).sum()
        # logger.debug('pot_sum = %f', pot_sum)
        return pot_sum

    @staticmethod
    def __normalize_pot_sigma(pot_sigma):
        pot_sigma_diff = pot_sigma - np.median(pot_sigma)
        exp = np.exp(pot_sigma_diff)
        normal = exp / exp.sum()
        if np.any(np.isnan(exp)) or np.any(np.isnan(normal)):
            logger.debug('pot_sigma_diff = %s', pot_sigma_diff)
            logger.debug('pot_sigma_diff max = %f', pot_sigma_diff.max())
            logger.debug('exp = %s', exp)
        return normal

    @classmethod
    def __sigma_dif(cls, sample, var, pred_indexes, child_indexes, lambdaa):
        """
        Return sigma(x1) - sigma(x0) where sigma(x) = sigma_{i=1..N} lambda_i * f(x) and x1 and x0 are equal to sample
        except that the variable "var" is changed to 0 and 1 respectively.
        :param sample:
        :param var:
        :param node_ids:
        :param pred_indexes:
        :return:
        """
        i, t = var
        # logger.debug('pred_indexes[i] = %s', pred_indexes[i])
        if t == 1:
            dif = lambdaa[i][sample[[i] + pred_indexes[i], 0]].sum()
        else:
            # logger.debug('child_indexes[i] = %s', child_indexes[i])
            dif = sum(lambdaa[j][1 + pred_indexes[j].index(i)] for j in child_indexes[i])
            if pred_indexes[i]:
                dif += lambdaa[i][0]
        return dif

    @classmethod
    def likelihood_gradients(cls, indexes, pred_indexes, child_indexes, data_size, feature_sums, lambdaa, c2,
                             gibbs_samples_num):
        return [
            cls.__likelihood_grad(i, pred_indexes, child_indexes, data_size, feature_sums[i], lambdaa, c2,
                                  gibbs_samples_num)
            for i in indexes
        ]

    @classmethod
    def __gibbs_samples_pot_sigma(cls, samples, pred_indexes, child_indexes, lambdaa):
        """
        Calculate potential sigma of samples given. The samples must be Gibbs samples in the order in which
        extracted at Gibbs sampling process.
        :param samples: list if N*2 arrays, each one, a sample from states space
        :param pred_indexes: dict of node indexes to their predecessor indexes
        :param child_indexes: dict of node indexes to their successor indexes
        :param lambdaa: list of lambda arrays
        :return: array of potential sigma values. The array size is equal to number of samples.
        """
        pot_sigma = [cls.__potentials_sigma(samples[0], pred_indexes, lambdaa)]
        last_sample = samples[0]
        last_sigma = pot_sigma[0]
        # logger.debug('samples[0].T = \n%s', two_d_arr_to_str(last_sample.T))
        # logger.debug('pot_sigma[0] = %f', last_sigma)

        for sample in samples[1:]:
            # logger.debug('sample.T = \n%s', two_d_arr_to_str(sample.T))
            if np.all(sample == last_sample):
                sigma = last_sigma
            else:
                nz = np.nonzero(sample != last_sample)
                i, t = nz[0][0], nz[1][0]
                # logger.debug('(i, t) = %s', (i, t))
                dif = cls.__sigma_dif(sample, (i, t), pred_indexes, child_indexes, lambdaa)
                # logger.debug('dif = %f', dif)
                if sample[i, t]:
                    sigma = last_sigma + dif
                else:
                    sigma = last_sigma - dif
                # logger.debug('sigma = %f', sigma)
            pot_sigma.append(sigma)
            last_sigma = sigma
            last_sample = sample

        return np.array(pot_sigma)

    @classmethod
    def __feat_i_by_sample(cls, sample, i, pred_indexes_i):
        pred_num = len(pred_indexes_i)
        return cls.__feature(sample[([i, i] + pred_indexes_i, [1] + [0] * (pred_num + 1))], pred_num)

    @classmethod
    def __gibbs_samples_feat_i(cls, samples, i, pred_indexes_i):
        """
        Extract features of node index i from the samples given. The samples must be Gibbs samples in the order in which
        extracted at Gibbs sampling process.
        :param samples: list if N*2 arrays, each one, a sample from states space
        :param i: the node index of which we want to extract feature
        :param pred_indexes_i: list of predecessor indexes of i
        :return: N*(d+1) matrix of features where N is number of samples and d is number of predecessors.
        """
        pred_num = len(pred_indexes_i)
        samples_num = len(samples)
        features = np.zeros((samples_num, pred_num + 1))
        features[0, :] = cls.__feat_i_by_sample(samples[0], i, pred_indexes_i)
        last_feat = features[0, :]
        last_sample = samples[0]
        # logger.debug('samples[0].T = \n%s', two_d_arr_to_str(last_sample.T))
        # logger.debug('features[0,:] = %s', arr_to_str(last_feat))
        # logger.debug('pred_indexes_i = %s', pred_indexes_i)

        for k in range(1, samples_num):
            sample = samples[k]
            # logger.debug('sample.T = \n%s', two_d_arr_to_str(sample.T))
            if np.all(sample == last_sample):
                features[k, :] = last_feat
            else:
                nz = np.nonzero(sample != last_sample)
                j, t = nz[0][0], nz[1][0]
                # logger.debug('(j, t) = %s', (j, t))
                if j == i:
                    if t == 1:
                        feat = sample[[i] + pred_indexes_i, 0].astype(float) if sample[i, 1] else np.zeros(
                            pred_num + 1)
                        features[k, :] = feat
                        last_feat = feat
                    else:
                        last_feat[0] = float(sample[i, t])
                        features[k, :] = last_feat
                elif j in pred_indexes_i:
                    last_feat[1 + pred_indexes_i.index(j)] = float(sample[i, t])
                    last_feat = last_feat
                else:
                    features[k, :] = last_feat

            # logger.debug('feat = %s', arr_to_str(features[k,:]))
            last_sample = sample

        return features


class Prediction:
    def __init__(self, parent_states, prob, state=True):
        self.parent_states = parent_states
        self.prob = prob
        self.state = state


def likelihood_grad(indexes, pred_indexes, child_indexes, data_size, feature_sums, lambdaa, c2, gibbs_samples_num):
    return UnifiedMRFModel.likelihood_gradients(indexes, pred_indexes, child_indexes, data_size, feature_sums,
                                                lambdaa, c2, gibbs_samples_num)
