import abc
import itertools
import pprint
import random
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from itertools import repeat
from typing import Sequence, List

import numpy as np
import psutil
from bson import ObjectId
from pympler.asizeof import asizeof

import settings
from cascade.models import CascadeTree
from db.exceptions import DataDoesNotExist
from db.managers import EvidenceManager, ParentSensEvidManager
from diffusion.models import DiffusionModel
from seq_labeling.unified_models import Prediction
from seq_labeling.utils import arr_to_str
from settings import logger
from utils.log_utils import log_memory
from utils.time_utils import time_measure, Timer


class SeqLabelDifModel(DiffusionModel, abc.ABC):
    """
    Sequence labeling diffusion model including HMM, MEMM, and CRF
    """

    def __init__(self, initial_depth, max_step, threshold, **kwargs):
        super().__init__(initial_depth, max_step, threshold, **kwargs)
        ''' self._models : a dictionary which its values are sequence labeling model instances and the keys are user 
        ids for receiver-based methods and tuples of (user_i, user_j) for sender-based methods.'''
        self._models = {}

    def _get_inactives(self, evidences):
        """
        Get totally inactive users which means they have no state 1.
        :type evidences:
        :return:
        :rtype:
        """
        user_ids = []
        for uid in evidences:
            for seq in evidences[uid]:
                if any(pair[1] for pair in seq):
                    break
            else:
                user_ids.append(uid)
        return user_ids

    @classmethod
    def train_models(cls, node_ids, trees, graph, iterations, project, eco=False, **kwargs):
        models = {}
        count = 0

        # logger.debug('calculating sizes ...')
        # size1, size2, size3 = asizeof(trees), asizeof(graph), asizeof(project)
        # logger.debug('size of trees : %d MB', size1 / 1024 ** 2)
        # logger.debug('size of graph : %d MB', size2 / 1024 ** 2)
        # logger.debug('size of project : %d MB', size3 / 1024 ** 2)
        # log_memory(locals(), globals())

        for node_id in node_ids:
            count += 1
            logger.debug('extracting evidences of %s ...', node_id)
            pred = graph.predecessors(node_id)
            node_ids = sorted(graph.nodes())
            sequences = cls.extract_node_evid(node_id, trees, pred, node_ids)

            if any(pair[1] != cls.inactive_state() for seq in sequences for pair in seq):
                # size4, size5 = asizeof(sequences), asizeof(models)
                # logger.debug('size of sequences : %d MB', size4 / 1024 ** 2)
                # logger.debug('size of models : %d MB', size5 / 1024 ** 2)
                # logger.debug('size sum = %d MB', (size1 + size2 + size3 + size4 + size5) / 1024 ** 2)
                # cls.log_node_evid(sequences)
                dim = sequences[0][0][0].size
                logger.debug('dim = %d, count = %d, dim * count = %d', dim, len(sequences), dim * len(sequences))
                logger.debug('training seq labeling model %d (user id: %s, dimensions: %d) ...', count, node_id,
                             dim)
                states = cls.get_states(len(pred))
                model = cls.train_model(sequences, iterations, states, node_id, project, eco, **kwargs)
                models[node_id] = model

            if count % 500 == 0 or count == len(node_ids):
                logger.info('%d models trained', count)

        return models

    def _fit_multiproc(self, node_ids, trees, iterations, graph, project, eco=False, **kwargs):
        """
        Train the seq labeling models using evidences given in multiprocessing mode.
        """
        random.shuffle(node_ids)

        futures = []
        step = min(500, len(node_ids) // settings.TRAIN_WORKERS + 1)
        with ProcessPoolExecutor(max_workers=settings.TRAIN_WORKERS) as executor:
            for i in range(0, len(node_ids), step):
                node_ids_i = node_ids[i:i + step]
                graph_i = graph.copy()
                graph_i.keep_node_ids(node_ids_i)
                f = executor.submit(train_models, type(self), node_ids_i, trees, graph_i, iterations, project, eco,
                                    **kwargs)
                futures.append(f)
        del graph_i

        logger.debug('assembling learned seq labeling models of processes ...')
        models = {}
        for f in futures:
            models.update(f.result())
        logger.debug('assembling done')

        return models

    @classmethod
    @abc.abstractmethod
    def train_model(cls, sequences: list, iterations, states, node_id, project, eco=False, **kwargs):
        pass

    def _fit_by_evidences(self, node_ids, train_trees, project, graph, iterations=None, multi_processed=False,
                          eco=False,
                          **kwargs):
        max_iteration = iterations if iterations is not None else self.max_iterations
        kwargs.update(self.get_params())
        manager = self._get_seq_label_manager(project)
        logger.info('training %d seq labeling models ...', graph.number_of_nodes())

        if multi_processed:
            models = self._fit_multiproc(node_ids, train_trees, max_iteration, graph, project, eco, **kwargs)
        else:
            logger.debug('training on single process ...')
            models = self.train_models(node_ids, train_trees, graph, max_iteration, project, eco, **kwargs)

        if eco and manager is not None:
            logger.info('inserting seq labeling models into db ...')
            manager.insert(models)

        return models

    @staticmethod
    def log_evidences(evidences):
        logs = ['evidences: ']
        for uid, evid in evidences.items():
            logs.append(f'uid: {uid}')
            logs.append(f'dimension: {evid[0][0][0].size}')
            logs.append('sequences:')
            for i in range(len(evid)):
                for obs, state in evid[i]:
                    logs.append(f'{arr_to_str(obs)} \t {state}')
                logs.append('')
        logger.debug('\n'.join(logs))

    @staticmethod
    def log_node_evid(sequences):
        logs = []
        for seq in sequences:
            for obs, state in seq:
                logs.append(f'{arr_to_str(obs)} \t {state}')
            logs.append('')
        logger.debug('evidences: \n%s', '\n'.join(logs))

    @time_measure(level='debug')
    def fit(self, train_set, train_trees, project, multi_processed=False, eco=False, iterations=None, **kwargs):
        """
        Load models from DB if existed, otherwise train a seq labeling model for each user in the training set.
        :return: self
        """
        super().fit(train_set, train_trees, project, multi_processed, eco)
        manager = self._get_seq_label_manager(project)

        # If it is in economical mode, train the models only if they are not saved in DB.
        if eco and manager is not None and manager.db_exists():
            logger.info('loading seq labeling models from db ...')
            self._models = manager.fetch_all()
        else:
            if eco and manager is not None and not manager.db_exists():
                logger.info('Seq labeling models do not exist in db.')
            node_ids = set(self.graph.nodes())
            # large_node_ids = {ObjectId('000000000000001713926427')}  # manual for Weibo Covid
            large_node_ids = set()
            not_large_node_ids = list(node_ids - large_node_ids)
            del node_ids
            logger.debug('num of node ids with large evidence: %d', len(large_node_ids))
            logger.debug('training models for not-large evidences ...')
            self._models = self._fit_by_evidences(not_large_node_ids, train_trees, project, self.graph, iterations,
                                                  multi_processed, eco, **kwargs)
            del not_large_node_ids

            if large_node_ids:
                logger.debug('training models for large evidences on single process ...')
                # Keep only the data for node ids with large evidence in self.graph to reduce memory usage.
                self.graph.keep_node_ids(large_node_ids)
                large_models = self._fit_by_evidences(large_node_ids, train_trees, project, self.graph, iterations,
                                                      multi_processed=False, eco=eco, **kwargs)
                self._models.update(large_models)

            logger.debug('memory usage: %f%%', psutil.virtual_memory()[2])
        return self

    @classmethod
    def _extract_obs(cls, cur_step_ids: list | set, obs_node_ids: list, timers=None) -> np.ndarray:
        """
        Get the observation array corresponding to the active node ids in the current step.
        :param cur_step_ids: current step node ids
        :param obs_node_ids: list of node ids related to the observation dimensions
        :return: the boolean observation array
        """
        # if timers is None:
        #     timers = [Timer(f'code {i}', level='debug') for i in range(10)]
        if not isinstance(cur_step_ids, set):
            cur_step_ids = set(cur_step_ids)
        # with timers[4]:
        new_active_indexes = [i for i in range(len(obs_node_ids)) if obs_node_ids[i] in cur_step_ids]
        dim = len(obs_node_ids)
        # with timers[5]:
        new_obs = np.zeros(dim, dtype=bool)
        # with timers[6]:
        new_obs[new_active_indexes] = 1
        return new_obs

    @abc.abstractmethod
    def _get_evid_manager(self, project):
        pass

    @classmethod
    @abc.abstractmethod
    def _get_seq_label_manager(cls, project):
        """
        Get an instance of SeqLabelDBManager that fits your Seq label model. Return None if you want to prevent
        reading models from and saving them into db.
        :param project: the project on which we train and test.
        """

    @classmethod
    @abc.abstractmethod
    def extract_node_evid(cls, node_id, trees, predecessors, all_node_ids=None):
        pass

    @staticmethod
    def inactive_state():
        return False

    @staticmethod
    def active_state(parent_id=None, predecessors=None):
        return True

    @staticmethod
    def get_states(pred_num=None):
        return [False, True]


def train_models(cls, node_ids, trees, graph, iterations, project, eco=False, **kwargs):
    try:
        return cls.train_models(node_ids, trees, graph, iterations, project, eco, **kwargs)
    except:
        logger.error(traceback.format_exc())
        raise


class NodeSeqLabelModel(SeqLabelDifModel, abc.ABC):
    """
    Node-based sequence labeling model
    """

    def _predict_one_thr(self, initial_tree, thr, graph, max_step=None):
        tree = initial_tree.copy()

        # Initialize values.
        max_depth = initial_tree.depth
        cur_step_nodes = initial_tree.nodes_at_depth(max_depth)  # Set the nodes with maximum depth as the initial step.
        cur_step = {node.user_id for node in cur_step_nodes}
        active_ids = set(initial_tree.node_ids())
        step_num = 1

        observations = self._get_initial_observations(initial_tree, cur_step_nodes, graph)
        # logger.debug('initial observations:\n%s', pprint.pformat(obs_dic))

        timers = [Timer(f'code {i}', level='debug', silent=True) for i in range(10)]

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

                # if child_id not in active_ids:
                model = self._get_model(child_id)

                if model is not None:
                    # logger.debug('testing diffusion to %s ...', child_id)
                    parents = list(graph.predecessors(child_id))

                    if child_id not in active_ids:
                        new_active_ids = cur_step & active_ids
                        obs_seq = observations.setdefault(child_id, [])
                        obs = self._extract_obs(new_active_ids, parents)

                        if obs_seq or np.any(obs):
                            obs_seq.append(obs)
                            # logger.debug('obs_seq = \n%s', [arr_to_str(obs) for obs in obs_seq])
                            parent_id, pred = self._predict_by_obs(obs_seq, thr, model, tree, parents)
                            if parent_id:
                                node = tree.add_node(child_id, parent_id=parent_id)
                                node.probability = pred.prob
                                active_ids.add(child_id)
                                next_step.add(child_id)

                    # else:
                    # logger.debug('user %s is already activated', child_id)

                # else:
                # logger.debug('user %s does not have any model', child_id)

                j += 1
                # if j % 100 == 0:
                # logger.debug('%d / %d of children iterated', j, len(all_children))
                # for timer in timers:
                #     if timer.sum:
                #         timer.report_sum()

            cur_step = next_step
            step_num += 1

        # logger.debug('size of tree: %d', asizeof(tree))
        # logger.debug('size of list: %d', asizeof(tree.edges()))
        return tree

    def predict_one_sample(self, initial_tree, threshold, graph, max_step=None):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree at each threshold.
        :return: dictionary of thresholds to their predicted trees
        """
        if not isinstance(threshold, list):
            return self._predict_one_thr(initial_tree, threshold, graph, max_step)

        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        trees = {thr: initial_tree.copy() for thr in threshold}

        # Initialize values.
        max_depth = initial_tree.depth
        cur_step_nodes = initial_tree.nodes_at_depth(max_depth)  # Set the nodes with maximum depth as the initial step.
        cur_step = {node.user_id for node in cur_step_nodes}
        init_nodes = set(initial_tree.node_ids())
        active_ids = {thr: init_nodes.copy() for thr in threshold}
        step_num = 1

        obs_dic = self._get_initial_observations(initial_tree, cur_step_nodes, graph)
        """
            Create dictionary of current observations of the nodes for each threshold:
            observations = {
                            threshold1: {user_id1: obs1, user_id2: obs2, ...},
                            threshold2: {user_id1: obs1, user_id2: obs2, ...},
                            ...
                            }
        """
        # logger.debug('initial observations:\n%s', pprint.pformat(obs_dic))
        observations = {thr: obs_dic.copy() for thr in threshold}

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

                # with timers[0]:
                model = self._get_model(child_id)

                if model is not None:
                    # logger.debug('testing diffusion to %s ...', child_id)
                    parents = list(graph.predecessors(child_id))
                    activated = False
                    last_pred = None

                    for thr in threshold:
                        thr_active_ids = active_ids[thr]
                        if child_id not in thr_active_ids:
                            # with timers[1]:
                            new_active_ids = cur_step & thr_active_ids
                            obs_seq = observations[thr].setdefault(child_id, [])
                            # with timers[2]:
                            obs = self._extract_obs(new_active_ids, parents)

                            if obs_seq or np.any(obs):
                                obs_seq.append(obs)
                                # if last_pred is None or not np.array_equal(obs, last_pred.obs):
                                #     logger.debug('obs = \n%s', arr_to_str(obs))
                                # logger.debug('threshold %f ...', thr)
                                # with timers[3]:
                                parent_id, pred = self._predict_by_obs(obs_seq, thr, model, trees[thr], parents,
                                                                       last_pred)
                                last_pred = pred
                                if parent_id:
                                    trees[thr].add_node(child_id, parent_id=parent_id)
                                    thr_active_ids.add(child_id)
                                    activated = True
                        # else:
                        # logger.debug('user %s is already activated', child_id)

                    if activated:
                        next_step.add(child_id)
                # else:
                # logger.debug('user %s does not have any model', child_id)

                j += 1
                # if j % 100 == 0:
                # logger.debug('%d / %d of children iterated', j, len(all_children))
                # for timer in timers:
                #     if timer.sum:
                #         timer.report_sum()

            cur_step = next_step
            step_num += 1

        return trees

    def _predict_parent_id(self, obs_seq, model, tree, obs_node_ids):
        # Set the parent with the maximum value of Lambda which is also activated at the current step as the
        # predicted parent of this child.
        conv_indexes = [model.orig_indexes_map.get(ind) for ind in obs_seq[0]]
        conv_indexes = list(filter(lambda x: x is not None, conv_indexes))
        if conv_indexes:
            max_lambda_ind = np.argmax(model.Lambda[conv_indexes])
            node_id = obs_node_ids[model.orig_indexes[conv_indexes[max_lambda_ind]]]
            if tree.get_node(node_id):
                return node_id
            else:
                logger.warning('parent node %s does not exist', node_id)
        # else:
        # logger.debug('the newly active nodes are not available in the training data')
        return

    def _predict_by_obs(self, obs_seq: List[np.ndarray], thr, model, tree, obs_node_ids, last_pred: Prediction = None):
        """

        :param obs_seq: observation sequence
        :param thr: threshold
        :param model: seq labeling model
        :param tree: current predicted tree
        :param obs_node_ids: list of node ids related to observation dimensions.
        :return: (parent_id, prediction instance) ; The parent id is returned if the diffusion is predicted,
        otherwise None.
        """
        if last_pred is not None and len(obs_seq) == len(last_pred.obs_seq) and all(
                np.array_equal(obs_seq[i], last_pred.obs_seq[i]) for i in range(obs_seq)):
            prob = last_pred.prob
        else:
            prob = model.get_prob(obs_seq, self.active_state(), self.get_states())
            # logger.debug('prob = %f', prob)
            if prob == np.nan:
                logger.warning('activation prob. of obs. %s is nan', obs_seq)
        pred = Prediction(obs_seq, prob)

        if prob >= thr:
            parent_id = self._predict_parent_id(obs_seq, model, tree, obs_node_ids)
            # if parent_id:
            # logger.debug('a diffusion predicted from %s with prob %f >= %f', parent_id, prob, thr)
            return parent_id, pred

        return None, pred

    def _get_initial_observations(self, initial_tree, max_depth_nodes, graph):
        observations = {}
        cur_step = initial_tree.roots
        max_depth_node_ids = set(node.user_id for node in max_depth_nodes)
        # logger.debug('max_depth_node_ids = %s', pprint.pformat(max_depth_node_ids))
        # logger.debug('extracting initial observations ...')

        i = 1
        while cur_step:
            # logger.debug('step %d with %d users ...', i, len(cur_step))
            # Just extract the observations of the node at the max depth of the initial tree.
            children_dic = {node.user_id: set(graph.successors(node.user_id)) & max_depth_node_ids for node in cur_step
                            if node.user_id in graph}
            cur_step_ids = [node.user_id for node in cur_step]
            # logger.debug('children_dic = %s', pprint.pformat(children_dic))
            parents_dic = {user_id: list(graph.predecessors(user_id)) for user_id in max_depth_node_ids if
                           user_id in graph}
            for node in cur_step:
                # logger.debug('node %s ...', node.user_id)
                children = children_dic.pop(node.user_id, [])  # to free RAM
                for child_id in children:
                    # logger.debug('child %s ...', child_id)
                    model = self._get_model(child_id)
                    if model is not None:
                        # Update the observation of this child.
                        parents = parents_dic[child_id]
                        observations.setdefault(child_id, [])
                        obs = self._extract_obs(cur_step_ids, parents)
                        observations[child_id].append(obs)
                        # logger.debug('obs set: %s', observations[child_id])

            next_step = reduce(lambda x, y: x + y, [node.children for node in cur_step], [])
            cur_step = next_step
            i += 1

        return observations

    def _get_evid_manager(self, project):
        return EvidenceManager(project)

    def _get_model(self, node_id):
        return self._models.get(node_id)


class Prediction:
    def __init__(self, obs_seq, prob, state=True):
        self.obs_seq = obs_seq
        self.prob = prob
        self.state = state


class MultiStateModel(NodeSeqLabelModel, abc.ABC):
    @staticmethod
    def inactive_state():
        return 0

    @staticmethod
    def active_state(parent_id=None, predecessors=None):
        return predecessors.index(parent_id) + 1

    @staticmethod
    def get_states(pred_num):
        return list(range(pred_num + 1))


class FullMultiStateModel(MultiStateModel, abc.ABC):
    """
    A MultiStateModel in which observations are of the length `nodes_count` and there is an (obs, state) per each
    active node added ordered by time.
    """

    @classmethod
    def extract_node_evid(cls, node_id, trees: List[CascadeTree], predecessors, all_node_ids=None):
        if all_node_ids is None:
            raise ValueError("all_node_ids must not be None")
        evidences = []  # list of sequences
        cascade_num = 1

        # Iterate each activation sequence and extract sequences of (observation, state) for the user
        for tree in trees:
            node = tree.get_node(node_id)
            act_depth = tree.depth_of(node_id) if node else None

            # if node is not None:
            #     logger.debug('node = %s', node)
            #     logger.debug('act_depth = %s', act_depth)

            sequence = []  # list of (obs, state) tuples for the current cascade
            for depth in range(tree.depth):
                if act_depth and depth >= act_depth:
                    break
                cur_step = sorted(tree.nodes_at_depth(depth), key=lambda x: x.datetime)
                cur_step_ids = [node.user_id for node in cur_step]
                sub_seq = []
                state = cls.inactive_state()
                finished = False
                for i in range(1, len(cur_step_ids) + 1):
                    if node and node.parent_id and cur_step_ids[i - 1] == node.parent_id:
                        state = cls.active_state(node.parent_id, predecessors)
                        finished = True
                    sub_seq.append((
                        cls._extract_obs(cur_step_ids[:i], all_node_ids),  # observation
                        state
                    ))
                    if finished:
                        break
                sequence.extend(sub_seq)

            # seq = "\n".join(f"obs: {''.join(str(int(d)) for d in obs)}\nstate: {state}" for obs, state in sequence)
            # logger.debug(f'sequence =\n{seq}')

            # if cascade_num % 100 == 0:
            #     logger.debug('%d cascades done', cascade_num)
            cascade_num += 1
            if sequence:
                evidences.append(sequence)

        return evidences

    def _predict_one_thr(self, initial_tree, thr, graph, max_step=None):
        tree = initial_tree.copy()

        # Initialize values.
        max_depth = initial_tree.depth
        # Set the nodes with maximum depth as the initial step.
        cur_step = sorted(initial_tree.nodes_at_depth(max_depth), key=lambda x: x.datetime)
        cur_step = [node.user_id for node in cur_step]
        # active_ids = set of node until last depth
        active_ids = set(initial_tree.node_ids()) - set(x.user_id for x in initial_tree.nodes_at_depth(max_depth))
        node_ids = sorted(graph.nodes())
        step_num = 1

        obs_seq = self._get_initial_obs(initial_tree, graph)
        # logger.debug('initial observations:\n%s', pprint.pformat(obs_seq))

        timers = [Timer(f'code {i}', level='debug', silent=True) for i in range(10)]

        # Predict the cascade tree.
        # At each iteration find newly activated nodes based on the probabilities and add them to the tree.
        while cur_step and (max_step is None or step_num <= max_step):
            # logger.debug('predicting step %d ...', step_num)
            next_step = []

            for node_id in cur_step:
                active_ids.add(node_id)

                if node_id in graph:
                    obs = self._extract_obs(active_ids, node_ids)
                    obs_seq.append(obs)

                    j = 0
                    for child_id in graph.successors(node_id):

                        model = self._get_model(child_id)

                        if model is not None:
                            # logger.debug('testing diffusion to %s ...', child_id)
                            parents = list(graph.predecessors(child_id))

                            if child_id not in active_ids:
                                # logger.debug('obs_seq = \n%s', [arr_to_str(obs) for obs in obs_seq])
                                parent_id, pred = self._predict_by_obs(obs_seq=obs_seq, thr=thr, model=model, tree=tree,
                                                                       predecessors=parents, node_ids=node_ids)
                                if parent_id:
                                    node = tree.add_node(child_id, parent_id=parent_id)
                                    node.probability = pred.prob
                                    active_ids.add(child_id)
                                    next_step.append(child_id)

                            # else:
                            #     logger.debug('user %s is already activated', child_id)

                        # else:
                        #     logger.debug('user %s does not have any model', child_id)

                        j += 1
                        # if j % 100 == 0:
                        #     logger.debug('%d / %d of children iterated', j, graph.out_degree(node_id))
                        #     for timer in timers:
                        #         if timer.sum:
                        #             timer.report_sum()

            cur_step = next_step
            step_num += 1

        # logger.debug('size of tree: %d', asizeof(tree))
        # logger.debug('size of list: %d', asizeof(tree.edges()))
        return tree

    def predict_one_sample(self, initial_tree, threshold, graph, max_step=None):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree at each threshold.
        :return: dictionary of thresholds to their predicted trees
        """
        if not isinstance(threshold, list):
            return self._predict_one_thr(initial_tree, threshold, graph, max_step)
        else:
            return {thr: self._predict_one_thr(initial_tree, thr, graph, max_step) for thr in threshold}

    def _get_initial_obs(self, initial_tree, graph):
        obs_seq = []
        node_ids = sorted(graph.nodes())
        # logger.debug('max_depth_node_ids = %s', pprint.pformat(max_depth_node_ids))
        # logger.debug('extracting initial observations ...')

        i = 1
        for depth in range(initial_tree.depth):
            # logger.debug('step %d with %d users ...', i, len(cur_step))
            # Just extract the observations of the node at the max depth of the initial tree.
            cur_step = sorted(initial_tree.nodes_at_depth(depth), key=lambda x: x.datetime)
            # logger.debug('children_dic = %s', pprint.pformat(children_dic))
            for i in range(1, len(cur_step) + 1):
                # logger.debug('node %s ...', node.user_id)
                obs = self._extract_obs(cur_step[:i], node_ids)
                obs_seq.append(obs)
                # logger.debug('obs set: %s', observations[child_id])
            i += 1

        return obs_seq

    def _predict_by_obs(self, obs_seq, thr, model, tree, predecessors, last_pred=None, node_ids=None):
        if last_pred is not None and len(obs_seq) == len(last_pred.obs_seq) and all(
                np.array_equal(obs_seq[i], last_pred.obs_seq[i]) for i in range(len(obs_seq))):
            state, prob = last_pred.state, last_pred.prob
        else:
            all_states = list(range(len(predecessors) + 1))
            probs = model.get_probs(obs_seq, all_states)
            new_act_indexes = np.nonzero(obs_seq[-1])[0]
            new_act_node_ids = [node_ids[ind] for ind in new_act_indexes]
            new_act_predec_indexes = [i for i in range(len(predecessors)) if predecessors[i] in new_act_node_ids]

            if new_act_predec_indexes:
                active_states = np.array(new_act_predec_indexes) + 1
                inactive_prob = probs[0]
                active_prob = 1 - inactive_prob
                active_probs = [probs[state] for state in active_states]
                i = np.argmax(active_probs)
                state, prob = active_states[i], active_prob
            else:
                state, prob = 0, 0
        pred = Prediction(obs_seq, prob, state)

        if state > 0 and prob >= thr:
            node_id = predecessors[state - 1]
            if tree.get_node(node_id):
                # logger.debug('a diffusion predicted from %s with prob %f >= %f', node_id, prob, thr)
                return node_id, pred
            else:
                logger.warning('parent node %s does not exist', node_id)

        return None, pred


class ParentSensMultiStateModel(MultiStateModel, abc.ABC):
    """
    A MultiStateModel in which observations are of the length `parents_count` and there is an (obs, state) per each
    tree step.
    """

    def _get_evid_manager(self, project):
        return ParentSensEvidManager(project)

    @classmethod
    def extract_node_evid(cls, node_id, trees, predecessors, all_node_ids=None):
        evidences = []  # list of sequences
        cascade_num = 1

        # Iterate each activation sequence and extract sequences of (observation, state) for the user
        for tree in trees:
            depth_to_pred = {}
            for pred_id in predecessors:
                pred_node = tree.get_node(pred_id)
                if pred_node:
                    depth = tree.depth_of(pred_id)
                    depth_to_pred.setdefault(depth, [])
                    depth_to_pred[depth].append(pred_id)
            # logger.debug('depth_to_pred = %s', depth_to_pred)

            node = tree.get_node(node_id)
            act_depth = tree.depth_of(node_id) if node else None
            # logger.debug('node = %s', node)
            # logger.debug('act_depth = %s', act_depth)

            sequence = []  # list of (obs, state) tuples for the current cascade
            for depth in sorted(depth_to_pred):
                if act_depth and depth >= act_depth:
                    break
                # Just the predecessors of the current step is sufficient to give as cur_step_ids.
                obs = cls._extract_obs(depth_to_pred[depth], predecessors)
                sequence.append((obs, cls.inactive_state()))
            # logger.debug('sequence = %s', sequence)

            if node and node.parent_id:
                sequence[-1] = (sequence[-1][0], cls.active_state(node.parent_id, predecessors))

            # if cascade_num % 100 == 0:
            #     logger.debug('%d cascades done', cascade_num)
            cascade_num += 1
            if sequence:
                evidences.append(sequence)

        return evidences

    def _predict_by_obs(self, obs_seq, thr, model, tree, obs_node_ids, last_pred=None):
        if last_pred is not None and len(obs_seq) == len(last_pred.obs_seq) and all(
                np.array_equal(obs_seq[i], last_pred.obs_seq[i]) for i in range(len(obs_seq))):
            state, prob = last_pred.state, last_pred.prob
        else:
            all_states = list(range(len(obs_node_ids) + 1))
            probs = model.get_probs(obs_seq, all_states)
            new_act_indexes = np.nonzero(obs_seq[-1])[0]
            if new_act_indexes.size != 0:
                active_states = new_act_indexes + 1
                inactive_prob = probs[0]
                active_prob = 1 - inactive_prob
                active_probs = [probs[state] for state in active_states]
                i = np.argmax(active_probs)
                state, prob = active_states[i], active_prob
            else:
                state, prob = 0, 0
        pred = Prediction(obs_seq, prob, state)

        if state > 0 and prob >= thr:
            node_id = obs_node_ids[state - 1]
            if tree.get_node(node_id):
                # logger.debug('a diffusion predicted from %s with prob %f >= %f', node_id, prob, thr)
                return node_id, pred
            else:
                logger.warning('parent node %s does not exist', node_id)

        return None, pred
