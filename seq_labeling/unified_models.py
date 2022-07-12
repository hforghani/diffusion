import itertools
import pprint
import typing
from functools import reduce

from matplotlib import pyplot as plt
import networkx
import numpy as np
from networkx import Graph
from scipy import sparse

from diffusion.enum import Method
from diffusion.models import DiffusionModel
from seq_labeling.utils import arr_to_str
from settings import logger
from utils.time_utils import Timer

timers = [Timer(f'code {i}', level='debug', silent=True) for i in range(10)]


def filter_none_values(seq):
    return list(filter(lambda x: x is not None, seq))


def merge(dict1, dict2):
    return dict(list(dict1.items()) + list(dict2.items()))


def report_sum():
    for timer in timers:
        if timer.sum:
            timer.report_sum()


class UnifiedMRFModel(DiffusionModel):
    method = Method.UNI_MRF
    max_iterations = 20

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, c2=0.1, eta=.1, epsilon=10 ** -6):
        super().__init__(initial_depth, max_step, threshold)
        self.c2 = c2
        self.eta = eta
        self.epsilon = epsilon
        self.__lambda = {}
        self.__nodes_map = {}

    def fit(self, train_set, train_trees, project, multi_processed=False, eco=False, **kwargs):
        super().fit(train_set, train_trees, project, multi_processed, eco)

        node_ids = list(self.graph.nodes())
        self.__nodes_map = {node_ids[i]: i for i in range(len(node_ids))}
        nodes_num = len(node_ids)
        max_depth = max(tree.depth for tree in train_trees)
        logger.debug('nodes_num = %d', nodes_num)
        logger.debug('max_depth = %d', max_depth)

        self.__lambda = self.__initialize_lambdaa(self.graph)
        states = self.__extract_states(train_trees, self.__nodes_map, max_depth)
        potential_sums = self.__calc_potential_sum(states, node_ids, self.__nodes_map, self.graph)
        junction_tree = self.__create_junction_tree(self.graph, node_ids, self.__nodes_map, max_depth)

        for i in range(self.max_iterations):
            logger.info('iteration %d', i + 1)
            dif_values = np.zeros(len(self.__lambda))
            new_lambda = {}
            j = 0
            for node_id in self.__lambda:
                dif = self.eta * self.__likelihood_grad(node_id, node_ids, self.__nodes_map, self.graph, states,
                                                        potential_sums[node_id], junction_tree)
                new_lambda[node_id] = self.__lambda[node_id] + dif
                dif_values[j] = np.linalg.norm(dif)
                j += 1
            self.__lambda = new_lambda
            # logger.debug('dif values = %s', arr_to_str(dif_values))
            logger.info('dif max = %f', np.max(np.abs(dif_values)))
            # logger.debug('dif sum = %f', np.sum(np.abs(dif_values)))
            if np.max(dif_values) < self.epsilon:
                logger.info('stop criterion met')
                break

        return self

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
            all_children = list(set().union(*children_sets))

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
        return {node_id: np.ones(graph.in_degree(node_id) + 1) for node_id in graph if graph.in_degree(node_id) > 0}

    def __likelihood_grad(self, node_id, node_ids, nodes_map, graph, states, potential_sum, junction_tree):
        cascades_num = len(states)  # = N
        max_depth = states[0].shape[1]  # = T
        i = nodes_map[node_id]
        pred_indexes = [nodes_map[j] for j in graph.predecessors(node_id)]
        lambdaa = self.__lambda[node_id]

        logger.info('calculating grad_ln_z ...')
        grad_ln_z = np.add.reduce([
            self.__joint_prob_i_t(i, t, values, pred_indexes, junction_tree, node_ids, nodes_map,
                                  graph) * self.__feature(i, t, values, pred_indexes)
            for t in range(2, max_depth + 1)
            for values in self.__state_values(i, t, pred_indexes)
        ])
        grad = potential_sum - cascades_num * grad_ln_z - lambdaa / self.c2
        return grad

    def __extract_states(self, train_trees, nodes_map, max_depth):
        """
        Extract the state matrices of the training data.
        :param train_trees: list of training set trees
        :param nodes_map: dict of node ids to node indexes
        :param max_depth: maximum depth of the training cascades.
        :return: list of N * T sparse matrices each corresponding to a cascade. N is number of nodes and T is the
        maximum depth. states_k[i,t] = 1 iff the node i is active at time t.
        """
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
        """
        Calculate the potential sum for each node id.
        :param states: list of training states matrices
        :param node_ids: list of training node ids
        :param nodes_map: dict of node ids to their indexes
        :param graph: underlying graph
        :return: dict of node ids to their potential sum terms appearing in the likelihood gradient formula.
        for node id "i", the potential sum is:
        \sum_{m=1}^M \sum_{t=1}^T f \left( {S_i^t}^{(m)}, {S_{Pa(i)}^{t-1}}^{(m)}\right)
        """
        logger.debug('calculating potential sums ...')
        potential_sums = {}
        cascades_num = len(states)
        nodes_num, max_depth = states[0].shape
        for node_id in node_ids:
            i = nodes_map[node_id]
            pred_indexes = [nodes_map[j] for j in graph.predecessors(node_id)]
            # logger.debug('node in-degree = %d', graph.in_degree(node_id))
            # logger.debug('pred_indexes = %s', pred_indexes)
            if pred_indexes:
                pot_sum_i = np.zeros(len(pred_indexes) + 1)
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
                    states_m = states[m]
                    pot_sum_i += np.add.reduce([
                        self.__feature(i, t,
                                       np.concatenate(
                                           (
                                               np.array([states_m[i, t - 2], states_m[i, t - 1]]),
                                               states_m[pred_indexes, t - 2].toarray().squeeze() if len(
                                                   pred_indexes) > 1 else np.array([states_m[pred_indexes[0], t - 2]])
                                           )
                                       ),
                                       pred_indexes)
                        for t in range(2, max_depth + 1)])
                potential_sums[node_id] = pot_sum_i
        # logger.debug('potential_sums = %s', pprint.pformat(potential_sums))
        return potential_sums

    def __state_array(self, states: dict, m: int, node_indexes: list, t: int) -> np.ndarray:
        if len(node_indexes) == 1:
            return np.array([states[m][node_indexes[0], t]])
        else:
            return states[m][node_indexes, t].toarray().squeeze()

    def __predict(self, thr, node_id, tree, parent_states, new_active_ids, parents, last_pred):
        if last_pred is not None and np.array_equal(parent_states, last_pred.parent_states):
            prob = last_pred.prob
        else:
            prob = self.__act_prob(parent_states, self.__lambda[node_id])
            logger.debug('prob = %f', prob)
        pred = Prediction(parent_states, prob)

        if prob >= thr:
            node_id = self.__predict_parent_id(node_id, new_active_ids, tree, parents)
            if node_id:
                logger.debug('a diffusion predicted from %s with prob %f >= %f', node_id, prob, thr)
            return node_id, pred

        return None, pred

    def __feature(self, i: int, t: int, values: np.ndarray, pred_indexes: list):
        """
        Get the feature vector f(S_i_t, S_Pa(i)_(t-1)).
        :param i:
        :param t:
        :param values: dict of tuples (i',t') to the values S_i'_t'
        :param pred_indexes: indexes of predecessors of node i
        :return: numpy array of the feature
        """
        # logger.debug('i, t = %s', (i, t))
        # logger.debug('values = %s', values)
        if values[0]:
            f = values[1:]
        else:
            f = np.zeros(values.size - 1, dtype=bool)
            # f = np.logical_not(np.array([values[i, t - 1]] + [values[(j, t - 1)] for j in pred_indexes]))
        # logger.debug('feature = %s', arr_to_str(f))
        return f

    def __act_prob(self, i, t, pred_indexes, pred_states, lambdaa):
        pred_values = merge({(j, t - 1): pred_states[j] for j in pred_indexes}, {(i, t - 1): False})
        f_true = self.__feature(i, t, merge(pred_values, {(i, t): True}), pred_indexes).astype(float)
        f_false = self.__feature(i, t, merge(pred_values, {(i, t): False}), pred_indexes).astype(float)
        # logger.debug('pred_states = %s', arr_to_str(pred_states))
        # logger.debug('lambdaa = %s', arr_to_str(lambdaa))
        # logger.debug('f_true = %s', arr_to_str(f_true))
        # logger.debug('f_false = %s', arr_to_str(f_false))
        return 1 / (1 + np.exp(lambdaa.dot(f_false - f_true)))

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
        lambdaa_new_act = self.__lambda[node_id][new_active_par_indicator]
        # logger.debug('lambdaa_new_act = %s', arr_to_str(lambdaa_new_act))
        max_lambda_ind = np.argmax(lambdaa_new_act)
        # logger.debug('max_lambda_ind = %s', max_lambda_ind)
        predicted_node_id = [p for p in parents if p in new_active_parents][max_lambda_ind]
        # logger.debug('predicted_node_id = %s', predicted_node_id)
        if tree.get_node(predicted_node_id):
            return predicted_node_id
        else:
            logger.warning('parent node %s does not exist', predicted_node_id)

    def __state_values(self, i: int, t: int, pred_indexes: list, fix_values: typing.Dict[tuple, object] = None) -> \
            typing.Generator[np.ndarray, None, None]:
        """
        Get the tuples of valid (S_i_t, S_Pa(i)_(t-1)) values. if fix_var is given, the value of fix_var is fixed to
        the given fix_val.
        :param i: nodex index i in S_i_t
        :param t: timestep t in S_i_t
        :param pred_indexes: indexes of predecessors of node with index i
        :param fix_values: If a dict of tuples (i',t') to values is given the variables S_i'_t' will be fixed to fix_values[(i',t')].
        :return: list of dicts of (i',t') to their values. Each dict is a unique value assignment.
        """
        # logger.debug('state values of %s with %d pred and fix variables %s', (i, t), len(pred_indexes), fix_values)
        i_values = [(False, False), (False, True), (True, True)]
        pred_val_ranges = [[False, True]] * len(pred_indexes)
        if fix_values:
            # logger.debug('fix_values = %s', fix_values)
            for var, value in fix_values.items():
                if len(var) != 2:
                    raise ValueError('length of each element of fix_vars must be 2')
                elif var == (i, t):
                    i_values = [val for val in i_values if val[1] == value]
                    # logger.debug('(i,t) fixed')
                elif var == (i, t - 1):
                    i_values = [val for val in i_values if val[0] == value]
                    # logger.debug('(i,t-1) fixed')
                elif var[0] in pred_indexes and var[1] == t - 1:
                    pred_val_ranges[pred_indexes.index(var[0])] = [value]
                    # logger.debug('(j,t-1) fixed')
                else:
                    raise ValueError(f'invalid value {value} for the click {(i, t)}')
        # logger.debug('i_values = %s', i_values)
        # logger.debug('pred_val_ranges = %s', pred_val_ranges)
        pred_values = itertools.product(*pred_val_ranges)
        num = np.prod(np.array([len(r) for r in pred_val_ranges])) * len(i_values)
        # logger.debug('iterating over %d values', num)
        counter = 0
        for pred_val in pred_values:
            for s_i_t_minus_1, s_i_t in i_values:
                counter += 1
                if counter % 10 ** 5 == 0:
                    logger.debug('%d%% of values done', counter / num * 100)
                    report_sum()
                with timers[0]:
                    yield np.array((s_i_t, s_i_t_minus_1) + pred_val)

    def __joint_prob_i_t(self, i, t, values_i_t, pred_indexes, junction_tree, node_ids, nodes_map, graph):
        """
        Compute the probability P(S_i^t, S_Pa(i)^(t-1)).
        :return:
        """
        root = (i, t)
        logger.debug('root = %s', root)
        logger.debug('values = %s', arr_to_str(values_i_t))

        # If the last state is active, the current step must be deterministically 1.
        if values_i_t[1]:  # S_i^{t - 1}
            return 1 if values_i_t[0] else 0

        bfs_tree = networkx.bfs_tree(junction_tree, root)
        leaves = {click for click in bfs_tree if bfs_tree.out_degree(click) == 0}
        # logger.debug('leaves = %s', leaves)
        del bfs_tree
        current_clicks = leaves.copy()
        next_clicks = set()
        messages = {}
        unmet = set(junction_tree.nodes())
        it = 0

        while current_clicks:
            logger.debug('current_clicks = %s', current_clicks)
            logger.debug('num of current_clicks = %d', len(current_clicks))
            logger.debug('num of unmet = %d', len(unmet))

            for src in current_clicks:
                unmet.remove(src)
                i_src, t_src = src
                src_pred_indexes = [nodes_map[j] for j in graph.predecessors(node_ids[i_src])]
                src_pred_ind_map = {src_pred_indexes[i]: i for i in range(len(src_pred_indexes))}
                node_id = node_ids[i_src]
                lambda_src = self.__lambda[node_id] if node_id in self.__lambda else 1
                # logger.debug('lambda of src = %s', arr_to_str(lambda_src))

                for dst in set(junction_tree.neighbors(src)) & unmet:
                    i_dst, t_dst = dst
                    if (src, dst) not in messages:
                        sep = src if t_src < t_dst else dst
                        messages[(src, dst)] = {}
                        for sep_val in [False, True]:
                            if src == sep and not sep_val:  # TODO: correct only if f=0 when S_i^t=0
                                src_values = []
                                src_values_num = 0
                            else:
                                src_values = self.__state_values(i_src, t_src, src_pred_indexes, fix_values={
                                    sep: sep_val,
                                    src: True  # TODO: correct only if f=0 when S_i^t=0
                                })
                                src_values_num = 2 ** len(src_pred_indexes)
                                if sep == src:
                                    src_values_num *= 2
                            logger.debug('calculating message of %s to %s (%s = %d)', src, dst, sep, sep_val)
                            logger.debug('num of predecessors = %d', len(src_pred_indexes))
                            logger.debug('num of src values = %d', src_values_num)

                            if src_values_num and src in leaves:
                                logger.debug('creating state values matrix ...')
                                with timers[4]:
                                    # TODO: correct only with binary/zero feature function.
                                    if not isinstance(lambda_src, np.ndarray):  # lambda = 1 and there is no predecessor
                                        m = src_values_num
                                    elif src == sep:
                                        m = src_values_num / 2 * np.sum(lambda_src)
                                    else:
                                        mul = src_values_num / 2 * lambda_src
                                        sep_index = 0 if sep == (i_src, t_src - 1) else src_pred_ind_map[sep[0]] + 1
                                        mul[sep_index] = sep_val * lambda_src[sep_index] * src_values_num
                                        m = np.sum(mul)
                            else:
                                m = 0  # message
                                src_neighbors = set(junction_tree.neighbors(src)) - {dst}
                                logger.debug('src neighbors except dst: %s', src_neighbors)
                                src_nei_msg = {(nei, val): messages[(nei, src)][val] for nei in src_neighbors for val in
                                               [False, True]}
                                for src_val in src_values:
                                    # logger.debug('src_val = %s', arr_to_str(src_val))
                                    with timers[1]:
                                        m_prod = np.multiply.reduce(
                                            [src_nei_msg[
                                                 (
                                                     nei,
                                                     self.__get_var_val(src_val,
                                                                        src if t_src < nei[1] else nei,
                                                                        i_src, t_src, src_pred_ind_map)
                                                 )
                                             ] for nei in src_neighbors]
                                        )
                                    with timers[2]:
                                        # TODO: correct only with binary/zero feature function.
                                        if isinstance(lambda_src, np.ndarray):
                                            # logger.debug('src_val[1:] = %s', src_val[1:])
                                            # logger.debug('lambda_src = %s', lambda_src)
                                            # logger.debug('src_val[1:] shape = %s', src_val[1:].shape)
                                            # logger.debug('lambda_src shape = %s', lambda_src.shape)
                                            m += np.sum(lambda_src[src_val[1:]]) * m_prod
                                        else:
                                            m += lambda_src * src_val[1] * m_prod
                            messages[(src, dst)][sep_val] = m
                            logger.debug('message of %s to %s (%s = %d) = %f', src, dst, sep, sep_val, m)

                    if len(set(junction_tree.neighbors(dst)) & unmet) <= 1:
                        next_clicks.add(dst)  # Means all messages have been received from the children.

            current_clicks = next_clicks
            next_clicks = set()
            it += 1
            logger.debug('%d iter of Shafer-Shenoy done', it)
            report_sum()

        logger.debug('summing up root prob ...')
        logger.debug('root neighbors: %s', list(junction_tree.neighbors(root)))
        pred_indexes_map = {pred_indexes[i]: i for i in range(len(pred_indexes))}
        m_prod = 1
        for nei in junction_tree.neighbors(root):
            i_nei, t_nei = nei
            sep = root if t < t_nei else nei
            m_prod *= messages[(nei, root)][self.__get_var_val(values_i_t, sep, i, t, pred_indexes_map)]
        prob = np.dot(self.__lambda[node_ids[i]], self.__feature(i, t, values_i_t, pred_indexes)) * m_prod
        logger.debug('prob of click %s = %f', root, prob)
        return prob

    def __create_junction_tree(self, graph, node_ids, nodes_map, max_depth):
        # Create click graph.
        nodes_num = len(nodes_map)
        logger.debug('creating click graph ...')
        click_graph = Graph()
        click_graph.add_edges_from(
            [((nodes_map[j], 2), (i, 3)) for i in range(nodes_num) for j in
             list(graph.predecessors(node_ids[i])) + [node_ids[i]]])
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
        logger.debug('creating junction tree ...')
        j_tree = mst.copy()
        j_tree.add_edges_from(
            [((i, t - 1), (i, t)) for i in range(nodes_num) for t in range(4, max_depth + 1)]
        )
        logger.debug('num of junction tree nodes, edges = %d, %d', j_tree.number_of_nodes(), j_tree.number_of_edges())
        # logger.debug('junction tree edges: \n%s', pprint.pformat(sorted(list(j_tree.edges()))))
        # logger.debug('drawing junction subtree ...')
        # subgraph = j_tree.subgraph([(i, t) for i in range(50) for t in range(2, max_depth + 1)])
        # pos = {n: (n[1] * 2 / max_depth - 1, 1 - n[0] * 2 / nodes_num) for n in subgraph.nodes()}
        # plt.figure(1, figsize=(200, 80), dpi=60)
        # networkx.draw_networkx(subgraph, pos)
        # plt.savefig('junction_tree.jpg', format="JPG")
        # logger.debug('done')
        return j_tree

    def __get_var_val(self, values, var, i, t, pred_indexes_map):
        """
        Get the value of the given variable in the array of click variables values.
        :param values: A P+2 length array of values of the click variables where P is number of predecessors. values[0]
        is S_i^t, values[1] is S_i^{t-1} and the rest are S_j^{t-1} for each j in predecessors of i.
        :param var: the query variable
        :param i: i of click
        :param t: t of click
        :param pred_indexes_map: dict of predecessor index to the index in Pa(i). i.e values[2+pred_indexes_map[j]] is
        related to S_j^{t-1}
        :return: the value of var
        """
        if var == (i, t):
            return values[0]
        elif var == (i, t - 1):
            return values[1]
        elif var[1] == t - 1:
            return values[2 + pred_indexes_map[var[0]]]
        else:
            raise ValueError('invalid query of var %s in click %s', var, (i, t))


class Prediction:
    def __init__(self, parent_states, prob, state=True):
        self.parent_states = parent_states
        self.prob = prob
        self.state = state
