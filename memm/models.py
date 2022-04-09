import collections
import logging
import math
import random
import traceback
from multiprocessing.pool import Pool

import psutil
from bson import ObjectId
from networkx import DiGraph
from pympler.asizeof import asizeof

import settings
from cascade.models import CascadeTree
from log_levels import DEBUG_LEVELV_NUM
from db.exceptions import DataDoesNotExist
from db.managers import MEMMManager, EvidenceManager, EdgeEvidenceManager, EdgeMEMMManager, ParentSensEvidManager
from diffusion.enum import Method
from memm.memm import *
from settings import logger
from utils.time_utils import Timer, time_measure


class Prediction:
    def __init__(self, obs, prob, state=True):
        self.obs = obs
        self.prob = prob
        self.state = state


class MEMMModel(abc.ABC):
    method = None  # Define in the subclasses.
    max_iterations = 500

    def __init__(self, project):
        self.project = project
        ''' self._memms : a dictionary which its values are MEMM instances and the keys are user ids for receiver-based 
        methods and tuples of (user_i, user_j) for sender-based methods.'''
        self._memms = {}
        self._last_obs = None
        self._last_prob = None
        self._last_state = None

    @time_measure(level='debug')
    def prepare_evidences(self, train_set, multi_processed=False):
        """
        Prepare the sequence of observations and states to train the MEMM models.
        :param train_set: list of training cascade id's
        :return: a dictionary which its values are MEMM evidences dictionaries with the format
                {'dimension': dimension, 'sequences': list_of_sequences}
                and the keys are user id's if the method is receiver-based and tuples of (user_i, user_j) if the
                method is sender-based.
        """
        logger.debug('method = %s', self.method)
        evid_manager = self._get_evid_manager()

        try:
            logger.info('loading MEMM evidences ...')
            evidences = evid_manager.get_many()
        except DataDoesNotExist:
            logger.info('no evidences found!')
            logger.info('Evidence extraction started')
            evidences = {}  # dictionary of user id's to list of the sequences of ObsPair instances.
            graph = self.project.load_or_extract_graph()
            trees = self.project.load_trees()
            more_args = self._more_args(graph)

            logger.info('extracting sequences from %d cascades ...', len(train_set))

            if multi_processed:
                process_count = min(settings.PROCESS_COUNT, len(train_set))
                pool = Pool(processes=process_count)
                step = int(math.ceil(float(len(train_set)) / process_count))
                results = []
                for j in range(0, len(train_set), step):
                    cascade_ids = train_set[j: j + step]
                    cur_trees = {cid: tree for cid, tree in trees.items() if cid in cascade_ids}
                    res = pool.apply_async(extract_evidences,
                                           args=(type(self), cascade_ids, graph, cur_trees),
                                           kwds=more_args)
                    results.append(res)

                pool.close()
                pool.join()

                logger.info('merging sequences of processes ...')
                for res in results:
                    process_evidences = res.get()
                    for key in process_evidences:
                        if key not in evidences:
                            evidences[key] = process_evidences[key]
                        else:
                            evidences[key]['sequences'].extend(process_evidences[key]['sequences'])

            else:
                evidences = self.extract_evidences(train_set, graph, trees, **more_args)

            # Delete evidences of totally inactive users since they will never be activated.
            inactives = self._get_inactives(evidences)
            for key in inactives:
                evidences.pop(key)
            logger.info('Evidences of %d totally inactive users deleted since they have no nonzero state',
                        len(inactives))

            # if settings.LOG_LEVEL <= DEBUG_LEVELV_NUM:
            #     logger.debugv('evidences = \n%s', pprint.pformat(evidences))

            logger.info('inserting %d evidences into db and creating indexes ...', len(evidences))
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
        put them in a dictionary named small_ev_keys. Put the others in a dictionary named large_ev_keys.
        :param evidences:
        :type evidences:
        :return:
        :rtype:
        """
        large_ev_keys = []
        small_ev_keys = []
        sizes = {}
        for key in evidences:
            sizes[key] = asizeof(evidences[key]['sequences'])
        sorted_uids = sorted(evidences.keys(), key=lambda uid: sizes[uid])
        size_sum = 0
        available = 0.8 * psutil.virtual_memory().available
        logger.debugv('available memory: %d G', available / 1024 ** 3)
        for key in sorted_uids:
            size_sum += sizes[key]
            if size_sum < available:
                small_ev_keys.append(key)
            else:
                large_ev_keys.append(key)
        # Shuffle user ids to balance the process memory sizes of processes (for small evidences).
        logger.debugv('num of small_ev_keys: %d', len(small_ev_keys))
        logger.debugv('size of 10 first small evidences: %s', [sizes[key] for key in small_ev_keys[:10]])
        logger.debugv('size of 10 first large evidences: %s', [sizes[key] for key in large_ev_keys[:10]])
        return large_ev_keys, small_ev_keys

    @classmethod
    def train_memms(cls, evidences, iterations, td_param=None, save_in_db=False, project=None):
        logger.debugv('training memms started')
        memms = {}
        count = 0
        manager = cls._get_memm_manager(project) if save_in_db else None
        graph = None
        if issubclass(cls, MultiStateMEMMModel):
            graph = project.load_or_extract_graph()

        for key, ev in evidences.items():
            count += 1
            logger.debug('training MEMM %d (user id: %s, dimensions: %d) ...', count, key, ev['dimension'])

            states = cls.get_states(key, graph)
            logger.debug('td_param = %s', td_param)
            memm = cls.get_memm_instance(td_param)
            logger.debug('type(memm) = %s', type(memm))

            try:
                memm.fit(ev, states, iterations)
                memms[key] = memm
            except MemmException:
                logger.warn('evidences for user %s ignored due to insufficient data', key)
            if count % 100 == 0:
                logger.info('%d memms trained', count)

            if save_in_db and count % 1000 == 0:
                logger.debug('inserting MEMMs into db ...')
                manager.insert(memms)
                memms = {}

        logger.debugv('training memms finished')
        if save_in_db and memms:
            logger.debug('inserting MEMMs into db ...')
            manager.insert(memms)
        else:
            return memms

    def _fit_multiproc(self, evidences, iterations, td_param=None):
        """
        Train the MEMMs using evidences given in multiprocessing mode.
        Side effect: Clears the evidences' dictionary.
        """
        keys = list(evidences.keys())
        random.shuffle(keys)
        process_count = min(settings.PROCESS_COUNT, len(evidences))
        logger.debug('starting %d processes to train MEMMs', process_count)
        pool = Pool(processes=process_count)
        step = int(math.ceil(len(evidences) / process_count))
        results = []

        for i in range(process_count):
            keys_i = keys[i * step: (i + 1) * step]
            evidences_i = {}
            for key in keys_i:
                evidences_i[key] = evidences.pop(key)  # to free RAM

            res = pool.apply_async(train_memms, (type(self), evidences_i, iterations, td_param, False, self.project))
            results.append(res)

        del evidences  # to free RAM
        pool.close()
        pool.join()
        memms = {}

        # Collect results of the processes.
        logger.debug('assembling learned MEMMs of processes ...')
        for res in results:
            memms_i = res.get()
            keys_i = list(memms_i.keys())
            for key in keys_i:
                memms[key] = memms_i.pop(key)  # to free RAM
        logger.debug('assembling done')

        return memms

    def _fit_by_evidences(self, train_set, iterations=None, td_param=None, multi_processed=False, eco=False):
        evidences = self.prepare_evidences(train_set, multi_processed)

        # if settings.LOG_LEVEL <= logging.DEBUG:
        #     self.log_evidences(evidences)

        memms = {}
        logger.info('training MEMMs started')

        max_iteration = iterations if iterations is not None else self.max_iterations
        logger.info('max iterations = %d', max_iteration)

        if multi_processed:
            single_process_ev = {}  # Evidences to train sequentially in a single process.
            multi_processed_ev = {}  # Evidences to train simultaneously in multiple processes.

            logger.info('separating large and small evidences ...')
            large_ev_keys, small_ev_keys = self._separate_big_ev(evidences)
            logger.info('%d large and %d small evidences considered', len(large_ev_keys), len(small_ev_keys))
            for key in large_ev_keys:
                single_process_ev[key] = evidences.pop(key)  # to free RAM
            for key in small_ev_keys:
                multi_processed_ev[key] = evidences.pop(key)  # to free RAM
            del evidences

            memms = self._fit_multiproc(multi_processed_ev, max_iteration, td_param)

            del multi_processed_ev
            if eco:
                logger.info('inserting MEMMs into db ...')
                manager = self._get_memm_manager(self.project)
                manager.insert(memms)
                memms = {}

        else:
            single_process_ev = evidences

        # Train big evidences sequentially in a single process if multi_processed is True and all evidences otherwise.
        logger.info('training %d MEMMs sequentially ...', len(single_process_ev))
        single_proc_memms = self.train_memms(single_process_ev, max_iteration, td_param, save_in_db=eco,
                                             project=self.project)
        del single_process_ev
        if not eco:
            memms.update(single_proc_memms)

        logger.info('training MEMMs finished')
        return memms

    def log_evidences(self, evidences):
        logs = ['evidences: ']
        for uid, evid in evidences.items():
            logs.append(f'uid: {uid}')
            logs.append(f'dimension: {evid["dimension"]}')
            logs.append(f'sequences:')
            for i in range(len(evid['sequences'])):
                logs.append(f'sequence {i}')
                for obs, state in evid[f'sequences'][i]:
                    logs.append(f'{obs_to_str(obs)} \t {state}\n')
        logger.debug('\n'.join(logs))

    @time_measure(level='debug')
    def fit(self, multi_processed=False, eco=False, iterations=None, td_param=None):
        """
        Load MEMM's from DB if exist, otherwise train a MEMM for each user in the training set.
        :return: self
        """
        train_set, _, _ = self.project.load_sets()
        manager = self._get_memm_manager(self.project)

        # If it is in economical mode, train the MEMMs only if they are not saved in DB.
        if not eco or not manager.db_exists():
            if eco and not manager.db_exists():
                logger.info('MEMMs do not exist in db.')
            if td_param:
                logger.info('TD parameter = %f', td_param)
            self._memms = self._fit_by_evidences(train_set, iterations, td_param, multi_processed, eco)

        if eco:
            logger.info('loading MEMMs from db ...')
            self._memms = manager.fetch_all()

        logger.debug('memory usage: %f%%', psutil.virtual_memory()[2])
        return self

    @abc.abstractmethod
    def predict(self, initial_tree: CascadeTree, graph: DiGraph, thresholds: list, max_step: int = None,
                **kwargs) -> dict:
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :return: dictionary of predicted tree for thresholds
        """

    def _fetch_children_parents(self, children_dic, graph):
        children = list(reduce(lambda x, y: x | y, (set(child_list) for child_list in children_dic.values()), set()))
        parents_dic = {user_id: list(graph.predecessors(user_id)) for user_id in children if user_id in graph}
        return parents_dic

    @classmethod
    def _update_obs(cls, obs: np.ndarray, cur_step_ids: collections.Iterable, obs_user_indexes: dict) -> np.ndarray:
        """
        Add a row to obs as the first row specifying current step active nodes.
        :param obs: current observation. None observation means there is no observation available.
        :param cur_step_ids: current step node ids
        :param obs_user_indexes: dictionary of node id to its column index in the observation array
        :return: a boolean showing whether an update is done, and the new observation array
        """
        new_active_indexes = [obs_user_indexes.get(uid) for uid in cur_step_ids]
        new_active_indexes = list(filter(lambda x: x is not None, new_active_indexes))
        dim = len(obs_user_indexes)
        first_row = np.zeros(dim, dtype=bool)
        if obs is None and not new_active_indexes:
            return None
        else:
            first_row[new_active_indexes] = 1
            new_obs = np.vstack((first_row, obs)) if obs is not None else first_row.reshape((1, dim))
            return new_obs

    def get_memm(self, key):
        if self._memms:
            return self._memms.get(key, None)
        else:
            return MEMMManager(self.project, self.method).fetch_one(key)

    def _more_args(self, graph):
        return {}

    @abc.abstractmethod
    def _get_evid_manager(self):
        pass

    @classmethod
    @abc.abstractmethod
    def _get_memm_manager(cls, project):
        pass

    @staticmethod
    @abc.abstractmethod
    def extract_evidences(train_set, graph, trees, **kwargs):
        pass

    @staticmethod
    def inactive_state():
        return False

    @staticmethod
    def active_state(node=None, graph=None):
        return True

    @staticmethod
    def get_states(key=None, graph=None):
        return [False, True]

    @staticmethod
    @abc.abstractmethod
    def get_memm_instance(td_param):
        pass


class NodeMEMMModel(MEMMModel, abc.ABC):
    """
    Node-based MEMM Model
    """

    @classmethod
    def extract_evidences(cls, train_set, graph, trees, **kwargs):
        logger.debug('NodeMEMMModel extract_memm_evidences started')
        try:
            ''' evidences: dictionary of user id's to dictionaries in the format 
                            {'dimension': dimension, 'sequences': list_of_sequences} '''
            evidences = {}
            cascade_num = 1

            # Iterate each activation sequence and extract sequences of (observation, state) for each user
            for cascade_id in train_set:
                cascade_seqs = {}  # dictionary of user id's to the sequences of ObsPair instances for the current cascade
                tree = trees[cascade_id]
                observations = {}  # current observation of each user
                activated = set()  # set of current activated users
                logger.debug('cascade %d ...', cascade_num)

                cur_step = tree.roots
                step_num = 1

                while cur_step:
                    logger.debug('step %d with %d users started', step_num, len(cur_step))
                    cur_step_ids = [node.user_id for node in cur_step]

                    # Put the state of the last observation in the sequence of (observation, state) equal to 1 (activated)
                    # for all nodes in the current step.
                    for node in cur_step:
                        uid = node.user_id
                        activated.add(uid)
                        parents_count = graph.in_degree(uid) if uid in graph else 0

                        if parents_count and uid in cascade_seqs and uid in observations:
                            if cascade_seqs[uid]:
                                obs = cascade_seqs[uid][-1][0]
                                state = cls.active_state(node, graph)
                                cascade_seqs[uid] = cascade_seqs[uid][:-1] + [(obs, state)]
                                logger.debugv('(obs, state) updated for %s : (%s, %d)', uid, obs, state)
                            del observations[uid]

                    # Get the children whom at least one of their parents are in the current step.
                    children_sets = (set(graph.successors(node_id)) for node_id in cur_step_ids if node_id in graph)
                    all_children = list(reduce(lambda x, y: x | y, children_sets, set()))
                    logger.debugv('all_children = %s', all_children)

                    # Update the observation of each child and add the new observation-state to the current sequences.
                    for child_id in all_children:
                        if child_id not in activated and child_id in graph:
                            obs = observations.get(child_id)
                            parents = list(graph.predecessors(child_id))
                            parent_indexes = {parents[i]: i for i in range(len(parents))}
                            new_obs = cls._update_obs(obs, cur_step_ids, parent_indexes)
                            observations[child_id] = new_obs
                            state = cls.inactive_state()
                            cascade_seqs.setdefault(child_id, [])
                            cascade_seqs[child_id].append((new_obs, state))
                            logger.debugv('(obs, state) added for %s : (%s, %d)', child_id, new_obs, state)

                    # Update the current step nodes.
                    cur_step = reduce(lambda li1, li2: li1 + li2, [node.children for node in cur_step])

                    logger.debug('%d steps done', step_num)
                    step_num += 1

                if cascade_num % 100 == 0:
                    logger.info('%d cascades done', cascade_num)
                cascade_num += 1

                # Add current sequence of pairs (observation, state) to the MEMM evidences.
                logger.debug('adding sequences of current cascade ...')
                if settings.LOG_LEVEL <= DEBUG_LEVELV_NUM:
                    logger.debugv('cascade_seqs =\n%s', pprint.pformat(cascade_seqs))
                for uid in cascade_seqs:
                    dim = graph.in_degree(uid)
                    evidences.setdefault(uid, {
                        'dimension': dim,  # TODO: Can I remove dimension?
                        'sequences': []
                    })
                    evidences[uid]['sequences'].append(cascade_seqs[uid])

            logger.info('evidences extracted from %d cascades', len(train_set))
            return evidences
        except:
            logger.error(traceback.format_exc())
            raise

    def _predict_one_thr(self, initial_tree, graph, thr, max_step=None):
        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        tree = initial_tree.copy()

        # Initialize values.
        max_depth = initial_tree.depth
        cur_step_nodes = initial_tree.nodes_at_depth(max_depth)  # Set the nodes with maximum depth as the initial step.
        cur_step = {node.user_id for node in cur_step_nodes}
        active_ids = set(initial_tree.node_ids())
        step_num = 1

        observations = self._get_initial_observations(initial_tree, cur_step_nodes, graph)
        # logger.debugv('initial observations:\n%s', pprint.pformat(obs_dic))

        timers = [Timer(f'code {i}', level='debug', silent=True) for i in range(10)]

        # Predict the cascade tree.
        # At each iteration find newly activated nodes based on MEMM probabilities and add them to the tree.
        while cur_step and (max_step is None or step_num <= max_step):
            logger.debug('predicting step %d ...', step_num)
            next_step = set()

            # Get the children whom at least one of their parents are in current step.
            children_sets = (set(graph.successors(node_id)) for node_id in cur_step if node_id in graph)
            all_children = list(reduce(lambda x, y: x | y, children_sets, set()))

            j = 0
            for child_id in all_children:

                # if child_id not in active_ids:
                memm = self.get_memm(child_id)

                if memm is not None:
                    logger.debugv('testing reshare to %s ...', child_id)
                    parents = list(graph.predecessors(child_id))
                    parents_map = {parents[i]: i for i in range(len(parents))}

                    if child_id not in active_ids:
                        new_active_ids = cur_step & active_ids
                        obs = observations.get(child_id)
                        obs = self._update_obs(obs, new_active_ids, parents_map)

                        if obs is not None:
                            observations[child_id] = obs
                            logger.debugv('obs = \n%s', obs_to_str(obs))
                            node_id, _ = self._predict_by_obs(obs, thr, memm, tree, parents)
                            if node_id:
                                tree.add_node(child_id, parent_id=node_id)
                                active_ids.add(child_id)
                                next_step.add(child_id)

                    else:
                        logger.debugv('user %s is already activated', child_id)

                else:
                    logger.debugv('user %s does not have any MEMM', child_id)

                j += 1
                if j % 100 == 0:
                    logger.debugv('%d / %d of children iterated', j, len(all_children))
                    # for timer in timers:
                    #     if timer.sum:
                    #         timer.report_sum()

            cur_step = next_step
            step_num += 1

        return tree

    def predict(self, initial_tree, graph, thresholds, max_step=None, **kwargs):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree at each threshold.
        :return: dictionary of thresholds to their predicted trees
        """
        if len(thresholds) == 1:
            tree = self._predict_one_thr(initial_tree, graph, thresholds[0], max_step)
            return {thresholds[0]: tree}

        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        trees = {thr: initial_tree.copy() for thr in thresholds}

        # Initialize values.
        max_depth = initial_tree.depth
        cur_step_nodes = initial_tree.nodes_at_depth(max_depth)  # Set the nodes with maximum depth as the initial step.
        cur_step = {node.user_id for node in cur_step_nodes}
        init_nodes = set(initial_tree.node_ids())
        active_ids = {thr: init_nodes.copy() for thr in thresholds}
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

        timers = [Timer(f'code {i}', level='debug', silent=True) for i in range(10)]

        # Predict the cascade tree.
        # At each iteration find newly activated nodes based on MEMM probabilities and add them to the tree.
        while cur_step and (max_step is None or step_num <= max_step):
            logger.debug('predicting step %d ...', step_num)

            next_step = set()

            # Get the children whom at least one of their parents are in current step.
            children_sets = (set(graph.successors(node_id)) for node_id in cur_step if node_id in graph)
            all_children = list(reduce(lambda x, y: x | y, children_sets, set()))

            j = 0
            for child_id in all_children:

                # if child_id not in active_ids:
                memm = self.get_memm(child_id)

                if memm is not None:
                    logger.debugv('testing reshare to %s ...', child_id)
                    with timers[1]:
                        parents = list(graph.predecessors(child_id))
                        parents_map = {parents[i]: i for i in range(len(parents))}
                    activated = False
                    last_pred = None

                    for thr in thresholds:
                        thr_active_ids = active_ids[thr]
                        if child_id not in thr_active_ids:
                            new_active_ids = cur_step & thr_active_ids
                            obs = observations[thr].get(child_id)
                            obs = self._update_obs(obs, new_active_ids, parents_map)

                            if obs is not None:
                                observations[thr][child_id] = obs
                                if not np.array_equal(obs, self._last_obs):
                                    logger.debugv('obs = \n%s', obs_to_str(obs))
                                logger.debugv('threshold %f ...', thr)
                                node_id, pred = self._predict_by_obs(obs, thr, memm, trees[thr], parents, last_pred)
                                last_pred = pred
                                if node_id:
                                    trees[thr].add_node(child_id, parent_id=node_id)
                                    thr_active_ids.add(child_id)
                                    activated = True
                        else:
                            logger.debugv('user %s is already activated', child_id)

                    if activated:
                        next_step.add(child_id)
                else:
                    logger.debugv('user %s does not have any MEMM', child_id)

                j += 1
                if j % 100 == 0:
                    logger.debugv('%d / %d of children iterated', j, len(all_children))
                    # for timer in timers:
                    #     if timer.sum:
                    #         timer.report_sum()

            cur_step = next_step
            step_num += 1

        return trees

    def _get_predicted_node_id(self, obs, memm, tree, obs_node_ids):
        # Set the parent with the maximum value of Lambda which is also activated at the current step as the
        # predicted parent of this child.
        conv_indexes = [memm.orig_indexes_map.get(ind) for ind in np.where(obs[0, :])[0]]
        conv_indexes = list(filter(lambda x: x is not None, conv_indexes))
        if conv_indexes:
            max_lambda_ind = np.argmax(memm.Lambda[conv_indexes])
            node_id = obs_node_ids[memm.orig_indexes[conv_indexes[max_lambda_ind]]]
            if tree.get_node(node_id):
                return node_id
            else:
                logger.warning('parent node %s does not exist', node_id)
        else:
            logger.debugv('the newly active nodes are not available in the training data')
        return

    def _predict_by_obs(self, obs, thr, memm, tree, obs_node_ids, last_pred=None):
        """

        :param obs: observation
        :param thr: threshold
        :param memm: MEMM
        :param tree: current predicted tree
        :param obs_node_ids: list of node ids related to observation dimensions.
        :return: The parent id is returned if the diffusion is predicted, otherwise None.
        """
        if last_pred is not None and np.array_equal(obs, last_pred.obs):
            prob = last_pred.prob
        else:
            prob = memm.get_prob(obs, self.active_state(), self.get_states())
            logger.debugv('prob = %f', prob)
            if prob == np.nan:
                logger.warning('activation prob. of obs. %s is nan', obs)
        pred = Prediction(obs, prob)

        if prob >= thr:
            node_id = self._get_predicted_node_id(obs, memm, tree, obs_node_ids)
            if node_id:
                logger.debugv('a reshare predicted from %s with prob %f >= %f', node_id, prob, thr)
            return node_id, pred

        return None, pred

    def _get_initial_observations(self, initial_tree, max_depth_nodes, graph):
        observations = {}
        cur_step = initial_tree.roots
        max_depth_node_ids = set(node.user_id for node in max_depth_nodes)
        # logger.debugv('max_depth_node_ids = %s', pprint.pformat(max_depth_node_ids))
        # logger.debugv('extracting initial observations ...')

        i = 1
        while cur_step:
            # logger.debugv('step %d with %d users ...', i, len(cur_step))
            # Just extract the observations of the node at the max depth of the initial tree.
            children_dic = {node.user_id: set(graph.successors(node.user_id)) & max_depth_node_ids for node in cur_step
                            if node.user_id in graph}
            cur_step_ids = [node.user_id for node in cur_step]
            # logger.debugv('children_dic = %s', pprint.pformat(children_dic))
            parents_dic = {user_id: list(graph.predecessors(user_id)) for user_id in max_depth_node_ids if
                           user_id in graph}
            for node in cur_step:
                # logger.debugv('node %s ...', node.user_id)
                children = children_dic.pop(node.user_id, [])  # to free RAM
                for child_id in children:
                    # logger.debugv('child %s ...', child_id)
                    memm = self.get_memm(child_id)
                    if memm is not None:
                        # Update the observation of this child.
                        parents = parents_dic[child_id]
                        parents_map = {parents[i]: i for i in range(len(parents))}
                        obs = observations.get(child_id)
                        obs = self._update_obs(obs, cur_step_ids, parents_map)
                        observations[child_id] = obs
                        # logger.debugv('obs set: %s', observations[child_id])

            next_step = reduce(lambda x, y: x + y, [node.children for node in cur_step], [])
            cur_step = next_step
            i += 1

        return observations

    def _get_evid_manager(self):
        return EvidenceManager(self.project)

    @classmethod
    def _get_memm_manager(cls, project):
        return MEMMManager(project, cls.method)


class EdgeMEMMModel(MEMMModel, abc.ABC):
    """
    Edge-based MEMM Model
    """

    max_iterations = 1000

    @classmethod
    def extract_evidences(cls, train_set, graph, trees, **kwargs):
        ''' evidences: dictionary of tuples of edges (user_id1, user_id2) to dictionaries in the format
                        {'dimension': dimension, 'sequences': list_of_sequences} '''
        logger.debug('EdgeMEMMModel extract_memm_evidences started')
        try:
            dim_user_indexes_map = kwargs['dim_user_indexes_map']
        except KeyError:
            raise ValueError('Keyword argument dim_user_indexes_map must be given')
        evidences = {}
        cascade_num = 1
        # timers = [Timer(f'predict part {i}', level='debug', unit=TimeUnit.SECONDS, silent=True) for i in range(10)]

        # Iterate each activation sequence and extract sequences of (observation, state) for each user
        for cascade_id in train_set:
            cascade_seqs = {}  # dictionary of edges to the sequences of tuples (observation, state) for the current cascade
            tree = trees[cascade_id]
            observations = {}  # current observation of each edge
            # activated = set()  # set of current activated edges
            logger.debug('cascade %d ...', cascade_num)

            cur_step = tree.roots
            step_num = 1

            while cur_step:
                logger.debug('step %d with %d users started', step_num, len(cur_step))
                cur_step_ids = [node.user_id for node in cur_step]
                cur_step_edges = [(node.parent_id, node.user_id) for node in cur_step if node.parent_id is not None]
                logger.debug('number of edges ending to current step: %d', len(cur_step_edges))

                # Put the state of the last observation in the sequence of (observation, state) equal to 1 (activated)
                # for all edges ending to the current step.
                for edge in cur_step_edges:
                    if edge in cascade_seqs:
                        obs = cascade_seqs[edge][-1][0]
                        cascade_seqs[edge] = cascade_seqs[edge][:-1] + [(obs, True)]
                        logger.debugv('(obs, state) updated for %s : (%s, %d)', edge, obs, True)
                        del observations[edge]

                # Extract the siblings and spouse edges of all nodes in the current step. The observations of these
                # nodes must be updated.
                sibling_edges = {(i, j) for node in cur_step for i in graph.predecessors(node.user_id) for j in
                                 graph.successors(i) if i != j}
                spouse_edges = {(i, j) for node in cur_step for j in graph.successors(node.user_id) for i in
                                graph.predecessors(j) if i != j}
                related_edges = list(sibling_edges | spouse_edges)

                # i = 0
                logger.debug('number of related edges to the current step: %d', len(related_edges))
                for edge in related_edges:
                    dim_users_map = dim_user_indexes_map[edge]
                    obs = observations.get(edge)
                    new_obs = cls._update_obs(obs, cur_step_ids, dim_users_map)
                    if new_obs:
                        observations[edge] = new_obs
                        cascade_seqs.setdefault(edge, [])
                        cascade_seqs[edge].append((new_obs, False))
                    # logger.debugv('(obs, state) added for %s : (%s, %d)', edge, new_obs, state)
                    # i += 1
                    # if i % 200 == 0:
                    #     for timer in timers:
                    #         if timer.sum != 0:
                    #             timer.report_sum()

                    # Update the current step nodes.
                    cur_step = reduce(lambda li1, li2: li1 + li2, [node.children for node in cur_step])

                    logger.debug('%d steps done', step_num)
                    step_num += 1

                    if cascade_num % 100 == 0:
                        logger.info('%d cascades done', cascade_num)
                    cascade_num += 1

                    # if settings.LOG_LEVEL <= DEBUG_LEVELV_NUM:
                    #     logger.debugv('cascade_seqs =\n%s', pprint.pformat(cascade_seqs))

                    # Add current sequence of pairs (observation, state) to the MEMM evidences.
                    logger.debug('adding sequences of current cascade ...')
                    for edge in cascade_seqs:
                        dim = len(dim_user_indexes_map[edge])
                    evidences.setdefault(edge, {
                        'dimension': dim,
                        'sequences': []
                    })
                    evidences[edge]['sequences'].append(cascade_seqs[edge])

                    logger.info('evidences extracted from %d cascades', len(train_set))
        return evidences

    def predict(self, initial_tree, graph, thresholds, max_step=None, dim_user_indexes_map=None):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :return: dictionary of thresholds to their predicted trees
        """
        if dim_user_indexes_map is None:
            raise ValueError('dim_user_indexes_map is required')

        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        trees = {thr: initial_tree.copy() for thr in thresholds}

        # Initialize values.
        max_depth = initial_tree.depth
        cur_step_nodes = initial_tree.nodes_at_depth(max_depth)  # Set the nodes with maximum depth as the initial step.
        cur_step = {node.user_id for node in cur_step_nodes}
        init_nodes = set(initial_tree.node_ids())
        active_ids = {thr: init_nodes.copy() for thr in thresholds}
        step_num = 1

        obs_dic = self._get_initial_observations(initial_tree, cur_step_nodes, graph, dim_user_indexes_map)
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

            next_step = set()

            for node_id in cur_step:

                if node_id not in graph:
                    logger.debug('node %s skipped since is not in graph', node_id)
                    continue

                children = list(graph.successors(node_id))
                logger.debug('predicting from node %s with %d children ...', node_id, len(children))

                j = 0
                for child_id in children:

                    edge = (node_id, child_id)
                    memm = self.get_memm(edge)

                    if memm is not None:
                        logger.debugv('testing reshare from %s to %s ...', node_id, child_id)

                        activated = False
                        last_prob, last_obs = None, None

                        for thr in thresholds:
                            thr_active_ids = active_ids[thr]
                            if child_id not in thr_active_ids:
                                index_map = dim_user_indexes_map[edge]
                                new_active_ids = cur_step & thr_active_ids
                                obs = observations[thr].get(edge)
                                obs = self._update_obs(obs, new_active_ids, index_map)

                                if obs is not None:
                                    observations[thr][edge] = obs
                                    if not np.array_equal(obs, last_obs):
                                        logger.debugv('obs = %s', obs_to_str(obs))
                                    logger.debugv('threshold %f ...', thr)
                                    res, prob = self._predict_by_obs(obs, edge, thr, memm, trees[thr], last_obs,
                                                                     last_prob)
                                    last_obs, last_prob = obs, prob
                                    if res:
                                        trees[thr].add_node(child_id, parent_id=node_id)
                                        thr_active_ids.add(child_id)
                                        activated = True

                        if activated:
                            next_step.add(child_id)
                    else:
                        logger.debugv('edge %s does not have any MEMM', edge)
                    # else:
                    #     logger.debugv('user %s is already activated', child_id)

                    j += 1
                    if j % 100 == 0:
                        logger.debugv('%d / %d of children iterated', j, len(children))

            cur_step = next_step
            step_num += 1

        return trees

    def _get_initial_observations(self, initial_tree, max_depth_nodes, graph, dim_user_indexes_map):
        observations = {}
        cur_step = initial_tree.roots
        max_depth_node_ids = {node.user_id for node in max_depth_nodes}
        graph_ids = set(graph.nodes())
        max_depth_edges = {(i, j) for i in max_depth_node_ids & graph_ids for j in graph.successors(i)}
        logger.debug('extracting initial observations ...')

        i = 1
        while cur_step:
            logger.debugv('step %d with %d users ...', i, len(cur_step))
            cur_step_ids = [node.user_id for node in cur_step]
            cur_step_ids_in_graph = set(cur_step_ids) & set(graph.nodes())

            # Extract the siblings and spouse edges of all nodes in the current step. The observations of these
            # nodes must be updated.
            sibling_edges = {(i, j) for uid in cur_step_ids_in_graph for i in graph.predecessors(uid) for j in
                             graph.successors(i) if i != j}
            spouse_edges = {(i, j) for uid in cur_step_ids_in_graph for j in graph.successors(uid) for i in
                            graph.predecessors(j) if i != j}
            # Just extract the observations of the edges after the max depth of the initial tree.
            related_edges = list((sibling_edges | spouse_edges) & max_depth_edges)
            logger.debugv('%d related edges', len(related_edges))

            for edge in related_edges:
                logger.debugv('edge (%s, %s)', edge[0], edge[1])
                index_map = dim_user_indexes_map[edge]
                obs = observations[edge]
                new_obs = self._update_obs(obs, cur_step_ids, index_map)
                observations[edge] = new_obs

            next_step = reduce(lambda x, y: x + y, [node.children for node in cur_step], [])
            cur_step = next_step
            i += 1

        if settings.LOG_LEVEL <= DEBUG_LEVELV_NUM:
            logger.debugv('initial observations =\n%s', pprint.pformat(observations))
        return observations

    def _more_args(self, graph):
        return dict(dim_user_indexes_map=self.extract_dim_user_indexes_map(graph))

    @staticmethod
    def extract_dim_user_indexes_map(graph):
        logger.info('extracting edge dimension users ...')
        edge_dim_users = {}
        for edge in graph.edges():
            src_children = graph.successors(edge[0])
            dest_parents = graph.predecessors(edge[1])
            edge_dim_users[edge] = sorted(set(src_children) | set(dest_parents) - {edge[1]})
        dim_user_indexes_map = {edge: {dim_users[i]: i for i in range(len(dim_users))} for edge, dim_users in
                                edge_dim_users.items()}
        return dim_user_indexes_map

    def __update_cur_step_obs(self, observations, cur_step_ids, graph, dim_user_indexes_map):
        # Extract the siblings and spouse edges of all nodes in the current step. The observations of these
        # nodes must be updated.
        sibling_edges = {(i, j) for uid in cur_step_ids for i in graph.predecessors(uid) for j in
                         graph.successors(i) if i != j}
        spouse_edges = {(i, j) for uid in cur_step_ids for j in graph.successors(uid) for i in
                        graph.predecessors(j) if i != j}
        related_edges = list(sibling_edges | spouse_edges)

        for edge in related_edges:
            index_map = dim_user_indexes_map[edge]
            obs = observations[edge]
            obs = self._update_obs(obs, cur_step_ids, index_map)
            observations[edge] = obs

    def _predict_by_obs(self, obs, edge, thr, memm, tree, last_obs=None, last_prob=None):
        if last_obs is not None and np.array_equal(obs, last_obs):
            prob = last_prob
        else:
            prob = memm.get_prob(obs, True, [False, True])
            logger.debugv('prob = %f', prob)
            if prob == np.nan:
                logger.warning('activation prob. of obs. %s is nan', obs)

        if prob >= thr:
            if tree.get_node(edge[0]):
                logger.debugv('a reshare predicted %f >= %f', prob, thr)
                return edge[0], prob
            else:
                logger.warning('parent node %s does not exist', edge[0])

        return None, prob

    def _get_evid_manager(self):
        return EdgeEvidenceManager(self.project)

    @classmethod
    def _get_memm_manager(cls, project):
        return EdgeMEMMManager(project, cls.method)


class MultiStateMEMMModel(NodeMEMMModel, abc.ABC):
    @staticmethod
    def inactive_state():
        return 0

    @staticmethod
    def active_state(node=None, graph=None):
        parents = list(graph.predecessors(node.user_id))
        index = parents.index(node.parent_id)
        return index + 1

    @staticmethod
    def get_states(key=None, graph=None):
        return list(range(graph.in_degree(key) + 1))

    def _get_evid_manager(self):
        return ParentSensEvidManager(self.project)

    def _predict_by_obs(self, obs, thr, memm, tree, obs_node_ids, last_pred=None):
        if last_pred is not None and np.array_equal(obs, last_pred.obs):
            state, prob = last_pred.state, last_pred.prob
        else:
            all_states = list(range(len(obs_node_ids) + 1))
            probs = memm.get_probs(obs, all_states)
            new_act_indexes = np.nonzero(obs[0, :])[0]
            if new_act_indexes.any():
                active_states = new_act_indexes + 1
                inactive_prob = probs[0]
                active_prob = 1 - inactive_prob
                active_probs = [probs[1 + i] for i in new_act_indexes]
                i = np.argmax(active_probs)
                state, prob = active_states[i], active_prob
            else:
                state, prob = 0, 0
        pred = Prediction(obs, prob, state)

        if state > 0 and prob >= thr:
            node_id = obs_node_ids[state - 1]
            if tree.get_node(node_id):
                logger.debugv('a reshare predicted from %s with prob %f >= %f', node_id, prob, thr)
                return node_id, pred
            else:
                logger.warning('parent node %s does not exist', node_id)

        return None, pred


class LongMEMMModel(NodeMEMMModel):
    method = Method.LONG_MEMM

    # max_iterations = 1000

    @staticmethod
    def get_memm_instance(td_param=None):
        return LongMEMM(td_param) if td_param is not None else LongMEMM()

    def _get_predicted_node_id(self, obs, memm, tree, obs_node_ids):
        # Set the parent with the maximum value of Lambda which is also activated at the current step as the
        # predicted parent of this child.
        obs_dim = len(obs_node_ids)
        conv_indexes = [memm.orig_indexes_map.get(ind) for ind in np.where(obs[0, :obs_dim])[0]]
        conv_indexes = list(filter(lambda x: x is not None, conv_indexes))
        if conv_indexes:
            max_lambda_ind = np.argmax(memm.Lambda[conv_indexes])
            node_id = obs_node_ids[memm.orig_indexes[conv_indexes[max_lambda_ind]]]
            if tree.get_node(node_id):
                return node_id
            else:
                logger.warning('parent node %s does not exist', node_id)
        else:
            logger.debugv('the newly active nodes are not available in the training data')

        return None


class MultiStateLongMEMMModel(LongMEMMModel, MultiStateMEMMModel):
    method = Method.MULTI_STATE_LONG_MEMM


class BinMEMMModel(NodeMEMMModel):
    method = Method.BIN_MEMM

    @staticmethod
    def get_memm_instance(td_param=None):
        return BinMEMM()


class MultiStateBinMEMMModel(BinMEMMModel, MultiStateMEMMModel):
    method = Method.MULTI_STATE_BIN_MEMM


class TDMEMMModel(NodeMEMMModel):
    """
    Time-Decay MEMM Model
    """
    method = Method.TD_MEMM

    @staticmethod
    def get_memm_instance(td_param=None):
        return TDMEMM(td_param) if td_param is not None else TDMEMM()


class MultiStateTDMEMMModel(TDMEMMModel, MultiStateMEMMModel):
    method = Method.MULTI_STATE_TD_MEMM

    @staticmethod
    def get_memm_instance(td_param=None):
        return TDMEMM(td_param) if td_param is not None else TDMEMM()


class ParentSensTDMEMMModel(MultiStateMEMMModel):
    """
    Parent-sensitive Time-Decay MEMM model
    """
    method = Method.PARENT_SENS_TD_MEMM

    @staticmethod
    def get_memm_instance(td_param=None):
        return ParentTDMEMM(td_param) if td_param is not None else ParentTDMEMM()


class LongParentSensTDMEMMModel(MultiStateMEMMModel):
    method = Method.LONG_PARENT_SENS_TD_MEMM

    @staticmethod
    def get_memm_instance(td_param=None):
        return LongParentTDMEMM(td_param) if td_param is not None else LongParentTDMEMM()


class TDEdgeMEMMModel(EdgeMEMMModel):
    method = Method.TD_EDGE_MEMM

    @staticmethod
    def get_memm_instance(td_param=None):
        return TDMEMM(td_param) if td_param is not None else TDMEMM()


def extract_evidences(cls, train_set, graph, trees, **kwargs):
    try:
        return cls.extract_evidences(train_set, graph, trees, **kwargs)
    except:
        logger.error(traceback.format_exc())
        raise


def train_memms(cls, evidences, iterations, td_param, save_in_db=False, project=None):
    try:
        return cls.train_memms(evidences, iterations, td_param, save_in_db, project)
    except:
        logger.error(traceback.format_exc())
        raise
