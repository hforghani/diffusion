import abc
import math
import pprint
import random
import typing
from abc import ABC
from functools import reduce
from multiprocessing.pool import Pool

import numpy as np
import psutil
import pymongo
from bson import ObjectId
from networkx import DiGraph
from pympler.asizeof import asizeof

import settings
from cascade.models import CascadeTree
from log_levels import DEBUG_LEVELV_NUM
from memm.asyncronizables import train_memms, extract_bin_memm_evidences, extract_reduced_memm_evidences, \
    divide_obs_by_2
from db.exceptions import DataDoesNotExist
from db.managers import MEMMManager, EvidenceManager
from db.reconnection import reconnect
from diffusion.enum import Method
from memm.memm import MEMM
from settings import logger
from utils.text_utils import columnize
from utils.time_utils import Timer, TimeUnit, time_measure


class MEMMModel(abc.ABC):
    method = None  # Define in the subclasses.

    def __init__(self, project):
        self.project = project
        self._memms = {}
        self._last_obs = None
        self._last_prob = None
        self._last_state = None
        self.max_iterations = 500

    @time_measure(level='debug')
    def _prepare_evidences(self, train_set, multi_processed=False, eco=False):
        """
        Prepare the sequence of observations and states to train the MEMM models.
        :param train_set: list of training cascade id's
        :return: a dictionary of user id's to instances of MemmEvidence
        """
        logger.debug('method = %s', self.method)
        evid_manager = EvidenceManager(self.project, self.method)

        evid_loaded = False
        if eco:
            try:
                logger.info('loading MEMM evidences ...')
                evidences = evid_manager.get_many()
                evid_loaded = True
            except DataDoesNotExist:
                logger.info('no evidences found!')

        if not evid_loaded:
            logger.info('Evidence extraction started')
            evidences = {}  # dictionary of user id's to list of the sequences of ObsPair instances.
            graph, act_seqs = self.project.load_or_extract_graph_seq()
            trees = self.project.load_trees()

            logger.info('extracting sequences from %d cascades ...', len(train_set))

            if multi_processed:
                process_count = min(settings.PROCESS_COUNT, len(train_set))
                pool = Pool(processes=process_count)
                step = int(math.ceil(float(len(train_set)) / process_count))
                results = []
                for j in range(0, len(train_set), step):
                    cascade_ids = train_set[j: j + step]
                    res = self._async_extract_evidences(pool, cascade_ids, graph, act_seqs, trees)
                    results.append(res)

                pool.close()
                pool.join()

                logger.info('merging sequences of processes ...')
                for res in results:
                    process_evidences = res.get()
                    for uid in process_evidences:
                        if uid not in evidences:
                            evidences[uid] = process_evidences[uid]
                        else:
                            evidences[uid]['sequences'].extend(process_evidences[uid]['sequences'])

            else:
                evidences = self._extract_evidences(train_set, graph=graph, act_seqs=act_seqs, trees=trees)

            # Delete evidences of totally inactive users since they will never be activated.
            inactives = self._get_inactives(evidences)
            for uid in inactives:
                evidences.pop(uid)
            logger.info('Evidences of %d totally inactive users deleted since they have no nonzero state',
                        len(inactives))

            if settings.LOG_LEVEL <= DEBUG_LEVELV_NUM:
                logger.debugv('evidences = \n%s', pprint.pformat(evidences))

            if eco:
                logger.info('inserting evidences into db and creating indexes ...')
                evid_manager.insert(evidences)
                evid_manager.create_index()

        return evidences

    def _get_inactives(self, evidences):
        """
        Get totally inactive users which means they have no state 1.
        :type evidences:
        :return:
        :rtype:
        """
        user_ids = []
        for uid in evidences:
            for seq in evidences[uid]['sequences']:
                if any(pair[1] for pair in seq):
                    break
            else:
                user_ids.append(uid)
        return user_ids

    def _separate_big_ev(self, evidences):
        """
        Sort the evidences by their sizes. Select as many small evidences to fill 80% of available memory and
        put them in a dictionary named small_ev_user_ids. Put the others in a dictionary named large_ev_user_ids.
        :param evidences:
        :type evidences:
        :return:
        :rtype:
        """
        large_ev_user_ids = []
        small_ev_user_ids = []
        sizes = {}
        for uid in evidences:
            sizes[uid] = asizeof(evidences[uid]['sequences'])
        sorted_uids = sorted(evidences.keys(), key=lambda uid: sizes[uid])
        size_sum = 0
        available = 0.8 * psutil.virtual_memory().available
        logger.debugv('available memory: %d G', available / 1024 ** 3)
        for uid in sorted_uids:
            size_sum += sizes[uid]
            if size_sum < available:
                small_ev_user_ids.append(uid)
            else:
                large_ev_user_ids.append(uid)
        # Shuffle user ids to balance the process memory sizes of processes (for small evidences).
        logger.debugv('num of small_ev_user_ids: %d', len(small_ev_user_ids))
        logger.debugv('size of 10 first small evidences: %s', [sizes[uid] for uid in small_ev_user_ids[:10]])
        logger.debugv('size of 10 first large evidences: %s', [sizes[uid] for uid in large_ev_user_ids[:10]])
        return large_ev_user_ids, small_ev_user_ids

    def _fit_multiproc(self, evidences, iterations):
        """
        Train the MEMMs using evidences given in multiprocessing mode.
        Side effect: Clears the evidences' dictionary.
        """
        user_ids = list(evidences.keys())
        random.shuffle(user_ids)
        process_count = min(settings.PROCESS_COUNT, len(evidences))
        logger.debug('starting %d processes to train MEMMs', process_count)
        pool = Pool(processes=process_count)
        step = int(math.ceil(len(evidences) / process_count))
        results = []

        for i in range(process_count):
            user_ids_i = user_ids[i * step: (i + 1) * step]
            evidences_i = {}
            for uid in user_ids_i:
                evidences_i[uid] = evidences.pop(uid)  # to free RAM

            # Train a MEMM for each user.
            res = pool.apply_async(train_memms, (evidences_i, self.method, iterations, False, self.project))
            results.append(res)

        del evidences  # to free RAM
        pool.close()
        pool.join()
        memms = {}

        # Collect results of the processes.
        logger.debug('assembling learned MEMMs of processes ...')
        for res in results:
            memms_i = res.get()
            user_ids_i = list(memms_i.keys())
            for uid in user_ids_i:
                memms[uid] = memms_i.pop(uid)  # to free RAM
        logger.debug('assembling done')

        return memms

    def _fit_by_evidences(self, train_set, multi_processed=False, eco=False, **kwargs):
        evidences = self._prepare_evidences(train_set, multi_processed, eco)
        memms = {}
        logger.info('training MEMMs started')

        max_iteration = kwargs.get('iterations', self.max_iterations)
        if max_iteration is None:
            max_iteration = self.max_iterations
        logger.info('max iterations = %d', max_iteration)

        if multi_processed:
            single_process_ev = {}  # Evidences to train sequentially in a single process.
            multi_processed_ev = {}  # Evidences to train simultaneously in multiple processes.

            logger.info('separating large and small evidences ...')
            large_ev_user_ids, small_ev_user_ids = self._separate_big_ev(evidences)
            logger.info('%d large and %d small evidences considered', len(large_ev_user_ids), len(small_ev_user_ids))
            for uid in large_ev_user_ids:
                single_process_ev[uid] = evidences.pop(uid)  # to free RAM
            for uid in small_ev_user_ids:
                multi_processed_ev[uid] = evidences.pop(uid)  # to free RAM
            del evidences

            memms = self._fit_multiproc(multi_processed_ev, max_iteration)

            del multi_processed_ev
            if eco:
                logger.info('inserting MEMMs into db ...')
                MEMMManager(self.project, self.method).insert(memms)
                memms = {}

        else:
            single_process_ev = evidences

        """
        Train big evidences sequentially in a single process if multi_processed is True and all
        evidences otherwise.
        """
        logger.info('training %d MEMMs sequentially ...', len(single_process_ev))
        single_proc_memms = train_memms(single_process_ev, self.method, max_iteration, save_in_db=True,
                                        project=self.project)
        del single_process_ev
        logger.debug('eco = %s', eco)
        logger.debug('single_proc_memms = %s', single_proc_memms)
        if not eco:
            memms.update(single_proc_memms)

        logger.info('training MEMMs finished')
        return memms

    @time_measure(level='debug')
    def fit(self, multi_processed=False, eco=False, **kwargs):
        """
        Load MEMM's from DB if exist, otherwise train a MEMM for each user in the training set.
        :return: self
        """
        train_set, _, _ = self.project.load_sets()
        manager = MEMMManager(self.project, self.method)

        # If it is in economical mode, train the MEMMs only if they are not saved in DB.
        if not eco or not manager.db_exists():
            if eco and not manager.db_exists():
                logger.info('MEMMs do not exist in db.')
            self._memms = self._fit_by_evidences(train_set, multi_processed, eco=eco, **kwargs)

        if eco:
            logger.info('loading MEMMs from db ...')
            try:
                self._memms = manager.fetch_all()

            except pymongo.errors.AutoReconnect:
                """ In the case of memory leak, it may raise AutoReconnect error. Then it loads MEMMs 
                one by one at the test stage via get_memm function. """
                logger.warning('AutoReconnect error!')
                reconnect()
            except MemoryError:
                """If the MEMMs size is too large, the Mongo connection will be lost due to memory leak.
                So it will be cleaned up. Then it fetches MEMMs one by one from db at the test stage via
                get_memm function."""
                logger.warning('Memory error!')
                reconnect()

        logger.debug('memory usage: %f%%', psutil.virtual_memory()[2])
        return self

    def predict(self, initial_tree: CascadeTree, graph: DiGraph, thresholds: list, max_step: int = None,
                multiprocessed: bool = True) -> dict:
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :return: dictionary of predicted tree for thresholds
        """
        timers = [Timer(f'predict part {i}', level='debug', unit=TimeUnit.SECONDS, silent=True) for i in range(10)]

        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        trees = {thr: initial_tree.copy() for thr in thresholds}

        # Initialize values.
        max_depth = initial_tree.depth
        cur_step_nodes = sorted(initial_tree.nodes_at_depth(max_depth),
                                key=lambda n: n.datetime)  # Set the nodes with maximum depth as the initial step.
        cur_step = [(node.user_id, set(thresholds)) for node in cur_step_nodes]
        active_ids = set(initial_tree.node_ids())
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
        observations = {thr: obs_dic.copy() for thr in thresholds}

        # Predict the cascade tree.
        # At each iteration find newly activated nodes based on MEMM probabilities and add them to the tree.
        while cur_step and (max_step is None or step_num <= max_step):
            logger.debug('predicting step %d ...', step_num)

            next_step = []

            cur_step_ids = [item[0] for item in cur_step]
            children_dic = {node_id: list(graph.successors(node_id)) for node_id in cur_step_ids if node_id in graph}
            parents_dic = self._fetch_children_parents(children_dic, graph)

            i = 0
            for node_id, node_thresholds in cur_step:
                children = children_dic.pop(node_id, [])  # to free RAM

                if children:
                    logger.debug('user %s has %d children:', node_id, len(children))
                    if settings.LOG_LEVEL <= DEBUG_LEVELV_NUM:
                        logger.debugv('\n' + columnize([str(child_id) for child_id in children], 4))

                    j = 0
                    for child_id in children:

                        if child_id not in active_ids:
                            memm = self._get_memm(child_id)

                            if memm is not None:

                                index = parents_dic[child_id].index(node_id)
                                child_thresholds = set()
                                last_prob = None
                                last_obs = None

                                for thr in thresholds:
                                    if thr in node_thresholds:
                                        with timers[1]:
                                            updated, _ = self._update_observation(child_id, [index], observations[thr],
                                                                                  memm)
                                        if updated:
                                            obs = observations[thr][child_id]
                                            logger.debugv('testing reshare to user %s using thr %f ...', child_id, thr)
                                            with timers[2]:
                                                if (obs == last_obs).all():
                                                    prob = last_prob
                                                else:
                                                    prob = memm.get_prob(obs, True, [False, True])
                                                    # prob = memm.get_prob(obs, True, [False, True], [timers[2], timers[3]])
                                                    if prob == np.nan:
                                                        logger.warning('activation prob. of obs. %s is nan', obs)
                                                    last_obs, last_prob = obs, prob

                                            if prob >= thr:
                                                if trees[thr].get_node(node_id):
                                                    trees[thr].add_node(child_id, parent_id=node_id)
                                                    child_thresholds.add(thr)
                                                    logger.debugv('a reshare predicted %f >= %f', prob, thr)
                                                else:
                                                    logger.warning('parent node %s does not exist', node_id)
                                    else:
                                        break

                                if child_thresholds:
                                    next_step.append((child_id, child_thresholds))
                                    active_ids.add(child_id)
                            else:
                                logger.debugv('user %s does not have any MEMM', child_id)
                        else:
                            logger.debugv('user %s is already activated', child_id)

                        j += 1
                        if j % 100 == 0:
                            logger.debugv('%d / %d of children iterated', j, len(children))

                i += 1
                logger.debug('%d / %d nodes of current step done', i, len(cur_step))
                if i % 200 == 0:
                    for timer in timers:
                        if timer.sum != 0:
                            timer.report_sum()

            for thr, obs_thr in observations.items():
                self._set_next_state_observations(obs_thr)

            cur_step = next_step
            step_num += 1

        for timer in timers:
            if timer.sum != 0:
                timer.report_sum()

        return trees

    def _update_observation(self, child_id: ObjectId, active_parent_indexes: list,
                            observations: typing.Dict[ObjectId, np.ndarray], child_memm: MEMM) -> typing.Tuple[
        bool, list]:
        """
        Update the observations of the child node for all thresholds in such a way that reflects the activation of the
        parents given.
        :param child_id: the child id
        :param active_parent_indexes: indexes of the newly active parents in the list of parents of the child
        :param observations: dictionary of child ids to their observations.
        :param child_memm: MEMM of the child node
        :return: tuple of (updated, conv_indexes). updated is True if the observation updated, False otherwise.
            conv_indexes is the list of original indexes of the current step active parents who are also in the
            training data.
        """
        updated = False
        obs = observations.setdefault(child_id, self._get_zero_obs(len(child_memm.orig_indexes)))
        conv_indexes = [child_memm.orig_indexes.index(ind) for ind in
                        set(active_parent_indexes) & set(child_memm.orig_indexes)]
        if conv_indexes:
            obs[conv_indexes] = 1
            updated = True
        return updated, conv_indexes

    def _get_zero_obs(self, dim):
        return np.zeros(dim)

    @abc.abstractmethod
    def _set_next_state_observations(self, observations: dict):
        """
        Prepare the observations for the next step.
        :param observations: dictionary of observations
        """

    def _fetch_children_parents(self, children_dic, graph):
        children = list(reduce(lambda x, y: x | y, (set(child_list) for child_list in children_dic.values()), set()))
        parents_dic = {user_id: list(graph.predecessors(user_id)) for user_id in children if user_id in graph}
        return parents_dic

    def _get_memm(self, user_id):
        if self._memms:
            return self._memms.get(user_id, None)
        else:
            return MEMMManager(self.project, self.method).fetch_one(user_id)

    def _get_initial_observations(self, initial_tree, max_depth_nodes, graph):
        observations = {}
        cur_step = initial_tree.roots
        max_depth_node_ids = set(node.user_id for node in max_depth_nodes)
        # logger.debugv('max_depth_node_ids = %s', pprint.pformat(max_depth_node_ids))
        # logger.debugv('extracting initial observations ...')

        i = 1
        while cur_step:
            # logger.debugv('step %d with %d users ...', i, len(cur_step))
            children_dic = {node.user_id: set(graph.successors(node.user_id)) & max_depth_node_ids for node in cur_step
                            if node.user_id in graph}
            # logger.debugv('children_dic = %s', pprint.pformat(children_dic))
            parents_dic = {user_id: list(graph.predecessors(user_id)) for user_id in max_depth_node_ids if
                           user_id in graph}
            for node in cur_step:
                # logger.debugv('node %s ...', node.user_id)
                children = children_dic.pop(node.user_id, [])  # to free RAM
                for child_id in children:
                    # logger.debugv('child %s ...', child_id)
                    memm = self._get_memm(child_id)
                    if memm is not None:
                        # Update the observation of this child.
                        index = parents_dic[child_id].index(node.user_id)
                        self._update_observation(child_id, [index], observations, memm)
                        # logger.debugv('obs set: %s', observations[child_id])

            next_step = reduce(lambda x, y: x + y, [node.children for node in cur_step], [])
            if next_step:
                self._set_next_state_observations(observations)
            cur_step = next_step
            i += 1

        return observations

    @abc.abstractmethod
    def _async_extract_evidences(self, pool, cascade_ids, graph, act_seqs, trees):
        pass

    @abc.abstractmethod
    def _extract_evidences(self, cascade_ids, graph, act_seqs, trees):
        pass


class ReducedMEMMModel(MEMMModel, ABC):
    def _async_extract_evidences(self, pool, cascade_ids, graph, act_seqs, trees):
        cur_trees = {cid: tree for cid, tree in trees.items() if cid in cascade_ids}
        res = pool.apply_async(extract_reduced_memm_evidences, (cascade_ids, graph, cur_trees, self.method))
        return res

    def _extract_evidences(self, cascade_ids, graph, act_seqs, trees):
        return extract_reduced_memm_evidences(cascade_ids, graph, trees, self.method)

    def predict(self, initial_tree, graph, thresholds, max_step=None, multiprocessed=True):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :return: dictionary of thresholds to their predicted trees
        """
        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        trees = {thr: initial_tree.copy() for thr in thresholds}

        # Initialize values.
        max_depth = initial_tree.depth
        cur_step_nodes = sorted(initial_tree.nodes_at_depth(max_depth),
                                key=lambda n: n.datetime)  # Set the nodes with maximum depth as the initial step.
        cur_step = [(node.user_id, set(thresholds)) for node in cur_step_nodes]
        active_ids = set(initial_tree.node_ids())
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
        # logger.debugv('initial observations:\n%s', pprint.pformat(obs_dic))
        observations = {thr: obs_dic.copy() for thr in thresholds}

        # Predict the cascade tree.
        # At each iteration find newly activated nodes based on MEMM probabilities and add them to the tree.
        while cur_step and (max_step is None or step_num <= max_step):
            logger.debug('predicting step %d ...', step_num)

            next_step = []

            # Get the children whom at least one of their parents are in current step.
            cur_step_ids = {item[0] for item in cur_step}
            children_sets = (set(graph.successors(node_id)) for node_id in cur_step_ids if node_id in graph)
            all_children = list(reduce(lambda x, y: x | y, children_sets, set()))

            cur_step_thresholds = {node_id: thr for node_id, thr in cur_step}

            j = 0
            for child_id in all_children:

                if child_id not in active_ids:
                    memm = self._get_memm(child_id)

                    if memm is not None:

                        parents = list(graph.predecessors(child_id))
                        cur_step_parents = set(parents) & cur_step_ids
                        child_thresholds = set()
                        self._last_prob = None
                        self._last_obs = None

                        for thr in thresholds:
                            parent_indexes = [parents.index(uid) for uid in cur_step_parents if
                                              thr in cur_step_thresholds[uid]]
                            if not parent_indexes:
                                break
                            updated, conv_indexes = self._update_observation(child_id, parent_indexes,
                                                                             observations[thr], memm)

                            if updated:
                                obs = observations[thr][child_id]
                                logger.debugv('testing reshare to user %s with obs %s using thr %f ...', child_id, obs,
                                              thr)
                                node_id = self._predict_by_obs(obs, thr, memm, trees[thr], parents, conv_indexes)
                                if node_id:
                                    trees[thr].add_node(child_id, parent_id=node_id)
                                    child_thresholds.add(thr)

                        # Set the maximum threshold in which each node is activated.
                        if child_thresholds:
                            next_step.append((child_id, child_thresholds))
                            active_ids.add(child_id)
                    else:
                        logger.debugv('user %s does not have any MEMM', child_id)
                else:
                    logger.debugv('user %s is already activated', child_id)

                j += 1
                if j % 100 == 0:
                    logger.debugv('%d / %d of children iterated', j, len(all_children))

            # Update the observations to prepare for the next step.
            for thr, obs_thr in observations.items():
                self._set_next_state_observations(obs_thr)

            cur_step = next_step
            step_num += 1

        return trees

    def _predict_by_obs(self, obs, thr, memm, tree, parents, conv_active_parent_indexes):
        """

        :param obs: observation
        :param thr: threshold
        :param memm: MEMM
        :param tree: current predicted tree
        :param parents: list of parents
        :param conv_active_parent_indexes: the indexes of the current step active parents in decreased dimension space
        of the MEMM features.
        :return: The parent id is returned if the diffusion is predicted, otherwise None.
        """
        if (obs == self._last_obs).all():
            prob = self._last_prob
        else:
            prob = memm.get_prob(obs, True, [False, True])
            if prob == np.nan:
                logger.warning('activation prob. of obs. %s is nan', obs)
            self._last_obs, self._last_prob = obs, prob

        if prob >= thr:
            # Set the parent with the maximum value of Lambda as the predicted parent of this child.
            max_lambda_ind = np.argmax(memm.Lambda[conv_active_parent_indexes])
            node_id = parents[memm.orig_indexes[conv_active_parent_indexes[max_lambda_ind]]]
            if tree.get_node(node_id):
                logger.debugv('a reshare predicted %f >= %f', prob, thr)
                return node_id
            else:
                logger.warning('parent node %s does not exist', node_id)

        return None


class BinMEMMModel(MEMMModel):
    method = Method.BIN_MEMM

    def _async_extract_evidences(self, pool, cascade_ids, graph, act_seqs, trees):
        cur_act_seqs = {cid: seq for cid, seq in act_seqs.items() if cid in cascade_ids}
        res = pool.apply_async(extract_bin_memm_evidences, (cascade_ids, graph, cur_act_seqs))
        return res

    def _extract_evidences(self, cascade_ids, graph, act_seqs, trees):
        return extract_bin_memm_evidences(cascade_ids, graph, act_seqs)

    def _get_zero_obs(self, dim):
        return np.zeros(dim, dtype=bool)

    def _set_next_state_observations(self, observations: dict):
        pass  # Do nothing!


class ReducedBinMEMMModel(ReducedMEMMModel):
    method = Method.REDUCED_BIN_MEMM

    def _get_zero_obs(self, dim):
        return np.zeros(dim, dtype=bool)

    def _set_next_state_observations(self, observations: dict):
        pass  # Do nothing!


class TDMEMMModel(MEMMModel):
    """
    Time-Decay MEMM Model
    """
    method = Method.TD_MEMM

    def _async_extract_evidences(self, pool, cascade_ids, graph, act_seqs, trees):
        cur_trees = {cid: tree for cid, tree in trees.items() if cid in cascade_ids}
        res = pool.apply_async(extract_reduced_memm_evidences, (cascade_ids, graph, cur_trees, self.method))
        return res

    def _extract_evidences(self, cascade_ids, graph, act_seqs, trees):
        return extract_reduced_memm_evidences(cascade_ids, graph, trees, self.method)

    def _set_next_state_observations(self, observations: typing.Dict[ObjectId, np.ndarray]):
        divide_obs_by_2(observations)


class ReducedTDMEMMModel(ReducedMEMMModel):
    """
    Time-Decay MEMM model with reduced observation-state sequences
    """
    method = Method.REDUCED_TD_MEMM

    def _set_next_state_observations(self, observations: typing.Dict[ObjectId, np.ndarray]):
        divide_obs_by_2(observations)


class ParentSensTDMEMMModel(ReducedTDMEMMModel):
    """
    Parent-sensitive Time-Decay MEMM model
    """
    method = Method.PARENT_SENS_TD_MEMM

    def _predict_by_obs(self, obs, thr, memm, tree, parents, conv_active_parent_indexes):
        if (obs == self._last_obs).all():
            state, prob = self._last_state, self._last_prob
        else:
            all_states = list(range(len(parents) + 1))
            probs = memm.get_probs(obs, all_states)
            active_states = [memm.orig_indexes[i] + 1 for i in conv_active_parent_indexes]
            inactive_prob = probs[0]
            active_prob = 1 - inactive_prob
            active_probs = [probs[1 + memm.orig_indexes[i]] for i in conv_active_parent_indexes]
            i = np.argmax(active_probs)
            state, prob = active_states[i], active_prob
            self._last_obs, self._last_prob, self._last_state = obs, prob, state

        if state > 0 and prob >= thr:
            node_id = parents[state - 1]
            if tree.get_node(node_id):
                logger.debugv('a reshare predicted from %s with prob %f >= %f', node_id, prob, thr)
                return node_id
            else:
                logger.warning('parent node %s does not exist', node_id)

        return None


class LongParentSensTDMEMMModel(ParentSensTDMEMMModel):
    method = Method.LONG_PARENT_SENS_TD_MEMM
