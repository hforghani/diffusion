import abc
import math
import pprint
import random
import typing
from functools import reduce
from multiprocessing.pool import Pool

import numpy as np
import psutil
import pymongo
from bson import ObjectId
from pymongo.errors import PyMongoError
from pympler.asizeof import asizeof

import settings
from cascade.models import CascadeTree
from log_levels import DEBUG_LEVELV_NUM
from memm.asyncronizables import train_memms, extract_bin_memm_evidences, extract_float_memm_evidences, \
    extract_parent_sens_float_memm_evidences
from db.exceptions import DataDoesNotExist
from db.managers import MEMMManager, EvidenceManager
from db.reconnection import reconnect
from memm.enum import MEMMMethod
from memm.memm import MEMM
from settings import logger
from utils.text_utils import columnize
from utils.time_utils import Timer, TimeUnit, time_measure


class MEMMModel(abc.ABC):
    method = None  # Define in the subclasses.

    def __init__(self, project):
        self.project = project
        self._memms = {}

    @time_measure(level='debug')
    def _prepare_evidences(self, train_set, multi_processed=False):
        """
        Prepare the sequence of observations and states to train the MEMM models.
        :param train_set: list of training cascade id's
        :return: a dictionary of user id's to instances of MemmEvidence
        """
        logger.debug('method = %s', self.method)
        evid_manager = EvidenceManager(self.project, self.method)

        try:
            logger.info('loading MEMM evidences ...')
            evidences = evid_manager.get_many()

        except DataDoesNotExist:
            logger.info('no evidences found! extraction started')
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
                    res = self._async_extract_evidences(pool, cascade_ids, graph=graph, act_seqs=act_seqs, trees=trees)
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
            logger.info('Evidences of %d totally inactive users deleted since they have no state 1', len(inactives))

            if settings.LOG_LEVEL <= DEBUG_LEVELV_NUM:
                logger.debugv('evidences = \n%s', pprint.pformat(evidences))

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

    def _fit_multiproc(self, evidences):
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
            res = pool.apply_async(train_memms, (evidences_i, self.method))
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

    def _fit_by_evidences(self, train_set, multi_processed=False):
        evidences = self._prepare_evidences(train_set, multi_processed)

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

            memms = self._fit_multiproc(multi_processed_ev)
            logger.info('inserting MEMMs into db ...')
            MEMMManager(self.project, self.method).insert(memms)
            del memms, multi_processed_ev

        else:
            single_process_ev = evidences

        """
        Train big evidences sequentially in a single process if multi_processed is True and all
        evidences otherwise.
        """
        logger.info('training %d MEMMs sequentially', len(single_process_ev))
        train_memms(single_process_ev, self.method, save_in_db=True, project=self.project)
        del single_process_ev

        logger.info('training MEMMs finished')

    @time_measure(level='debug')
    def fit(self, train_set, multi_processed=False):
        """
        Load MEMM's from DB if exist, otherwise train MEMM's for each user in training set.
        :param train_set:   cascade id's in training set
        :return:            self
        """
        manager = MEMMManager(self.project, self.method)

        # Train MEMMs if they are not saved in DB.
        if not manager.db_exists():
            logger.info('MEMMs do not exist in db. They will be trained')
            self._fit_by_evidences(train_set, multi_processed)

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

    def predict(self, initial_tree: CascadeTree, thresholds: list, max_step: int = None,
                multiprocessed: bool = True) -> dict:
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :return: dictionary of predicted tree for thresholds
        """
        graph = self.project.load_or_extract_graph()
        timers = [Timer(f'predict part {i}', level='debug', unit=TimeUnit.SECONDS, silent=True) for i in range(10)]

        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        trees = {thr: initial_tree.copy() for thr in thresholds}

        # Initialize values.
        max_depth = initial_tree.depth
        cur_step_nodes = sorted(initial_tree.nodes_at_depth(max_depth),
                                key=lambda n: n.datetime)  # Set the nodes with maximum depth as the initial step.
        max_thr = max(thresholds)
        cur_step = [(node.user_id, max_thr) for node in cur_step_nodes]
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
            for node_id, max_predicted_thr in cur_step:
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
                                child_max_pred_thr = None
                                last_prob = None
                                last_obs = None

                                for thr in thresholds:
                                    if thr <= max_predicted_thr:
                                        with timers[1]:
                                            updated = self._update_observation(child_id, [index], observations[thr],
                                                                               memm)
                                        if updated:
                                            obs = observations[thr][child_id]
                                            logger.debugv('testing reshare to user %s using thr %f ...', child_id, thr)
                                            with timers[2]:
                                                if (obs == last_obs).all():
                                                    prob = last_prob
                                                else:
                                                    prob = memm.get_prob(obs)
                                                    # prob = memm.get_prob(obs, [timers[2], timers[3]])
                                                    if prob == np.nan:
                                                        logger.warning('activation prob. of obs. %s is nan', obs)
                                                    last_obs, last_prob = obs, prob

                                            if prob >= thr:
                                                if trees[thr].get_node(node_id):
                                                    trees[thr].add_child(node_id, child_id)
                                                    child_max_pred_thr = thr
                                                    logger.debugv('a reshare predicted %f >= %f', prob, thr)
                                                else:
                                                    logger.warning('parent node %s does not exist', node_id)
                                    else:
                                        break

                                if child_max_pred_thr is not None:
                                    next_step.append((child_id, child_max_pred_thr))
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
                            observations: typing.Dict[ObjectId, np.ndarray], child_memm: MEMM) -> bool:
        """
        Update the observations of the child node for all thresholds in such a way that reflects the activation of the
        parent given.
        :param child_id: the child id
        :param active_parent_indexes: index of the newly active parent in the list of parents of the child
        :param observations: dictionary of child ids to their observations.
        :param child_memm: MEMM of the child node
        :return: True if the observation updated, False otherwise.
        """
        updated = False
        obs = observations.setdefault(child_id, self._get_zero_obs(len(child_memm.orig_indexes)))
        for ind in active_parent_indexes:
            try:
                converted_ind = child_memm.orig_indexes.index(ind)
            except ValueError:
                pass
            else:
                obs[converted_ind] = 1
                updated = True
        return updated

    @abc.abstractmethod
    def _get_zero_obs(self, dim):
        pass

    @abc.abstractmethod
    def _set_next_state_observations(self, observations: dict):
        """
        Divide all observations by 2 in order to prepare for the next step.
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
    def _async_extract_evidences(self, pool, cascade_ids, **kwargs):
        pass

    @abc.abstractmethod
    def _extract_evidences(self, cascade_ids, **kwargs):
        pass


class BinMEMMModel(MEMMModel):
    method = MEMMMethod.BIN_MEMM

    def _async_extract_evidences(self, pool, cascade_ids, **kwargs):
        act_seqs = kwargs.get('act_seqs')
        graph = kwargs.get('graph')
        if act_seqs is None:
            raise ValueError('keyword argument "act_seqs" must be given')
        if graph is None:
            raise ValueError('keyword argument "graph" must be given')
        cur_act_seqs = {cid: seq for cid, seq in act_seqs.items() if cid in cascade_ids}
        res = pool.apply_async(extract_bin_memm_evidences, (cascade_ids, graph, cur_act_seqs))
        return res

    def _extract_evidences(self, cascade_ids, **kwargs):
        act_seqs = kwargs.get('act_seqs')
        graph = kwargs.get('graph')
        if act_seqs is None:
            raise ValueError('keyword argument "act_seqs" must be given')
        if graph is None:
            raise ValueError('keyword argument "graph" must be given')
        return extract_bin_memm_evidences(cascade_ids, graph, act_seqs)

    def _get_zero_obs(self, dim):
        return np.zeros(dim, dtype=bool)

    def _set_next_state_observations(self, observations: dict):
        pass  # Do nothing!


class FloatMEMMModel(MEMMModel):
    method = MEMMMethod.FLOAT_MEMM

    def _async_extract_evidences(self, pool, cascade_ids, **kwargs):
        trees = kwargs.get('trees')
        graph = kwargs.get('graph')
        if trees is None:
            raise ValueError('keyword argument "trees" must be given')
        if graph is None:
            raise ValueError('keyword argument "graph" must be given')
        cur_trees = {cid: tree for cid, tree in trees.items() if cid in cascade_ids}
        res = pool.apply_async(extract_float_memm_evidences, (cascade_ids, graph, cur_trees))
        return res

    def _extract_evidences(self, cascade_ids, **kwargs):
        trees = kwargs.get('trees')
        graph = kwargs.get('graph')
        if trees is None:
            raise ValueError('keyword argument "trees" must be given')
        if graph is None:
            raise ValueError('keyword argument "graph" must be given')
        return extract_float_memm_evidences(cascade_ids, graph, trees)

    def _get_zero_obs(self, dim):
        return np.zeros(dim)

    def _set_next_state_observations(self, observations: typing.Dict[ObjectId, np.ndarray]):
        for user_id, obs in observations.items():
            observations[user_id] = obs / 2


class ParentFloatMEMMModel(FloatMEMMModel):
    method = MEMMMethod.PARENT_SENS_FLOAT_MEMM

    def _async_extract_evidences(self, pool, cascade_ids, **kwargs):
        trees = kwargs.get('trees')
        graph = kwargs.get('graph')
        if trees is None:
            raise ValueError('keyword argument "trees" must be given')
        if graph is None:
            raise ValueError('keyword argument "graph" must be given')
        cur_trees = {cid: tree for cid, tree in trees.items() if cid in cascade_ids}
        res = pool.apply_async(extract_parent_sens_float_memm_evidences, (cascade_ids, graph, cur_trees))
        return res

    def _extract_evidences(self, cascade_ids, **kwargs):
        trees = kwargs.get('trees')
        graph = kwargs.get('graph')
        if trees is None:
            raise ValueError('keyword argument "trees" must be given')
        if graph is None:
            raise ValueError('keyword argument "graph" must be given')
        return extract_parent_sens_float_memm_evidences(cascade_ids, graph, trees)

    def predict(self, initial_tree, thresholds, max_step=None, multiprocessed=True):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :return: dictionary of predicted tree for thresholds
        """
        graph = self.project.load_or_extract_graph()

        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        trees = {thr: initial_tree.copy() for thr in thresholds}

        # Initialize values.
        max_depth = initial_tree.depth
        cur_step_nodes = sorted(initial_tree.nodes_at_depth(max_depth),
                                key=lambda n: n.datetime)  # Set the nodes with maximum depth as the initial step.
        thr = max(thresholds)
        cur_step = [(node.user_id, thr) for node in cur_step_nodes]
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
        logger.debugv('initial observations:\n%s', pprint.pformat(obs_dic))
        observations = {thr: obs_dic.copy() for thr in thresholds}

        # Predict the cascade tree.
        # At each iteration find newly activated nodes based on MEMM probabilities and add them to the tree.
        while cur_step and (max_step is None or step_num <= max_step):
            logger.debug('predicting step %d ...', step_num)

            next_step = []

            cur_step_ids = {item[0] for item in cur_step}
            children_dic = {node_id: list(graph.successors(node_id)) for node_id in cur_step_ids if node_id in graph}
            all_children = list(
                reduce(lambda x, y: x | y, (set(child_list) for child_list in children_dic.values()), set()))
            parents_dic = {user_id: list(graph.predecessors(user_id)) for user_id in all_children if user_id in graph}
            cur_step_max_thr = {node_id: thr for node_id, thr in cur_step}

            j = 0
            for child_id in all_children:

                if child_id not in active_ids:
                    memm = self._get_memm(child_id)

                    if memm is not None:

                        cur_step_parents = set(parents_dic[child_id]) & cur_step_ids
                        parents = parents_dic[child_id]
                        cur_step_parents_max_thr = {uid: thr for uid, thr in cur_step_max_thr.items() if
                                                    uid in cur_step_parents}
                        child_max_pred_thr = None

                        for thr in thresholds:
                            parent_indexes = [parents.index(uid) for uid in cur_step_parents if
                                              thr <= cur_step_parents_max_thr[uid]]
                            if not parent_indexes:
                                break
                            updated, conv_indexes = self._update_observation(child_id, parent_indexes,
                                                                             observations[thr], memm)
                            obs = observations[thr][child_id]

                            # Test all possible parent-sensitive observations and find their maximum probability.
                            logger.debugv('testing reshare to user %s using thr %f ...', child_id, thr)
                            probs = []
                            for ind in conv_indexes:
                                par_sens_obs = obs.copy()
                                par_sens_obs[ind] = 1
                                prob = memm.get_prob(par_sens_obs)
                                logger.debugv('obs = %s, prob = %s', par_sens_obs, prob)
                                if prob == np.nan:
                                    logger.warning('activation prob. of obs. %s is nan', par_sens_obs)
                                probs.append(prob)
                            probs = np.array(probs)
                            max_prob = np.max(probs)

                            # If the maximum probability is greater than the threshold, activate it and add it to the
                            # tree at this threshold.
                            if max_prob >= thr:
                                max_ind = np.argmax(probs)
                                node_id = parents[parent_indexes[max_ind]]
                                if trees[thr].get_node(node_id):
                                    trees[thr].add_child(node_id, child_id)
                                    child_max_pred_thr = thr
                                    par_sens_obs = obs.copy()
                                    par_sens_obs[max_ind] = 1
                                    observations[thr][child_id] = par_sens_obs
                                    logger.debugv('a reshare predicted %f >= %f', max_prob, thr)
                                else:
                                    logger.warning('parent node %s does not exist', node_id)

                        # Set the maximum threshold in which each node is activated.
                        if child_max_pred_thr is not None:
                            next_step.append((child_id, child_max_pred_thr))
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

    def _update_observation(self, child_id, active_parent_indexes, observations, child_memm):
        obs = observations.setdefault(child_id, self._get_zero_obs(len(child_memm.orig_indexes)))
        conv_indexes = []
        updated = False
        for ind in active_parent_indexes:
            try:
                converted_ind = child_memm.orig_indexes.index(ind)
            except ValueError:
                conv_indexes.append(None)
            else:
                obs[converted_ind] = 0.5
                conv_indexes.append(converted_ind)
                updated = True
        return updated, conv_indexes
