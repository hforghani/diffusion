import abc
import collections
import math
import pprint
import random
import traceback
from functools import reduce
from multiprocessing.pool import Pool

import numpy as np
import psutil
from pympler.asizeof import asizeof

import settings
from db.exceptions import DataDoesNotExist
from db.managers import EvidenceManager, ParentSensEvidManager
from diffusion.models import DiffusionModel
from log_levels import DEBUG_LEVELV_NUM
from seq_labeling.utils import obs_to_str
from settings import logger
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

    @time_measure(level='debug')
    def prepare_evidences(self, train_set, train_trees, project, graph, multi_processed=False, eco=False):
        """
        Prepare the sequence of observations and states to train the sequence labeling models.
        :param train_set: list of training cascade id's
        :return: a dictionary which its values are the evidences dictionaries in the format
                {'dimension': dimension, 'sequences': list_of_sequences}
                and the keys are user id's if the method is receiver-based and tuples of (user_i, user_j) if the
                method is sender-based.
        """
        logger.debug('method = %s', self.method)

        must_extract = True
        evidences = {}
        if eco:
            evid_manager = self._get_evid_manager(project)
            try:
                logger.info('loading evidences ...')
                evidences = evid_manager.get_many()
                must_extract = False
            except DataDoesNotExist:
                logger.info('no evidences found!')

        if must_extract:
            logger.info('Evidence extraction started')
            evidences = {}  # dictionary of user id's to list of the sequences of ObsPair instances.
            more_args = self._more_args(graph)

            logger.info('extracting sequences from %d cascades ...', len(train_set))

            if multi_processed:
                process_count = min(settings.PROCESS_COUNT, len(train_set))
                pool = Pool(processes=process_count)
                step = int(math.ceil(float(len(train_set)) / process_count))
                results = []
                for j in range(0, len(train_set), step):
                    cascade_ids = train_set[j: j + step]
                    cur_trees = train_trees[j: j + step]
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
                evidences = self.extract_evidences(train_trees, graph, **more_args)

            # Delete evidences of totally inactive users since they will never be activated.
            inactives = self._get_inactives(evidences)
            for key in inactives:
                evidences.pop(key)
            logger.info('Evidences of %d totally inactive users deleted since they have no nonzero state',
                        len(inactives))

            # if settings.LOG_LEVEL <= DEBUG_LEVELV_NUM:
            #     logger.debugv('evidences = \n%s', pprint.pformat(evidences))

            if eco:
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
    def train_models(cls, evidences, iterations, graph, save_in_db=False, project=None, **kwargs):
        logger.info('training %d seq labeling models ...', len(evidences))
        logger.debug('kwargs = %s', kwargs)
        models = {}
        count = 0
        manager = cls._get_seq_label_manager(project) if save_in_db else None

        for key, ev in evidences.items():
            count += 1
            logger.debug('training seq labeling model %d (user id: %s, dimensions: %d) ...', count, key,
                         ev['dimension'])

            model = cls._train_model(ev, iterations, key, graph, **kwargs)
            models[key] = model

            if count % 100 == 0:
                logger.debug('%d models trained', count)

            if save_in_db and manager is not None and count % 1000 == 0:
                logger.debug('inserting seq labeling models into db ...')
                manager.insert(models)

        if save_in_db and manager is not None and models:
            logger.debug('inserting seq labeling models into db ...')
            manager.insert(models)

        logger.info('done')
        return models

    def _fit_multiproc(self, evidences, iterations, project, graph, **kwargs):
        """
        Train the seq labeling models using evidences given in multiprocessing mode.
        Side effect: Clears the evidences' dictionary.
        """
        keys = list(evidences.keys())
        random.shuffle(keys)
        process_count = min(settings.PROCESS_COUNT, len(evidences))
        logger.debug('starting %d processes to train seq labeling models', process_count)
        pool = Pool(processes=process_count)
        step = int(math.ceil(len(evidences) / process_count))
        results = []

        for i in range(process_count):
            keys_i = keys[i * step: (i + 1) * step]
            evidences_i = {}
            for key in keys_i:
                evidences_i[key] = evidences.pop(key)  # to free RAM

            res = pool.apply_async(train_models, (type(self), evidences_i, iterations, graph, False, project),
                                   kwargs)
            results.append(res)

        del evidences  # to free RAM
        pool.close()
        pool.join()
        models = {}

        # Collect results of the processes.
        logger.debug('assembling learned seq labeling models of processes ...')
        for res in results:
            models_i = res.get()
            keys_i = list(models_i.keys())
            for key in keys_i:
                models[key] = models_i.pop(key)  # to free RAM
        logger.debug('assembling done')

        return models

    @classmethod
    @abc.abstractmethod
    def _train_model(cls, evidence, iterations, key, graph, **kwargs):
        pass

    def _fit_by_evidences(self, train_set, train_trees, project, graph, iterations=None, multi_processed=False,
                          eco=False, **kwargs):
        logger.debug('kwargs = %s', kwargs)
        evidences = self.prepare_evidences(train_set, train_trees, project, graph, multi_processed, eco)
        # if settings.LOG_LEVEL <= logging.DEBUG:
        #     self.log_evidences(evidences)

        models = {}
        max_iteration = iterations if iterations is not None else self.max_iterations
        kwargs.update(self.get_params())
        manager = self._get_seq_label_manager(project)
        logger.info('training seq labeling models ...')

        if multi_processed:
            single_process_ev = {}  # Evidences to train sequentially in a single process.
            multi_processed_ev = evidences

            models = self._fit_multiproc(multi_processed_ev, max_iteration, project, graph, **kwargs)

            del multi_processed_ev
            if eco and manager is not None:
                logger.info('inserting seq labeling models into db ...')
                manager.insert(models)
        else:
            single_process_ev = evidences

        if single_process_ev:
            # Train big evidences sequentially in a single process if multi_processed is True and all evidences
            # otherwise.
            logger.info('training %d seq labeling models sequentially ...', len(single_process_ev))
            single_proc_models = self.train_models(single_process_ev, max_iteration, graph, save_in_db=eco,
                                                   project=project, **kwargs)
            del single_process_ev
            if not eco or manager is None:
                models.update(single_proc_models)

        return models

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
    def fit(self, train_set, train_trees, project, multi_processed=False, eco=False, iterations=None, **kwargs):
        """
        Load models from DB if existed, otherwise train a seq labeling model for each user in the training set.
        :return: self
        """
        logger.debug('kwargs = %s', kwargs)
        logger.info('params = %s', self.get_params())
        self.project = project
        graph = project.load_or_extract_graph(train_set)
        self.graph = graph

        manager = self._get_seq_label_manager(project)

        # If it is in economical mode, train the models only if they are not saved in DB.
        if eco and manager is not None and manager.db_exists():
            logger.info('loading seq labeling models from db ...')
            self._models = manager.fetch_all()
        else:
            if eco and manager is not None and not manager.db_exists():
                logger.info('Seq labeling models do not exist in db.')
            self._models = self._fit_by_evidences(train_set, train_trees, project, graph, iterations, multi_processed,
                                                  eco, **kwargs)

        logger.debug('memory usage: %f%%', psutil.virtual_memory()[2])
        return self

    @classmethod
    def _update_obs(cls, obs: np.ndarray, cur_step_ids: collections.Iterable, obs_node_ids: list,
                    timers=None) -> np.ndarray:
        """
        Add a row to obs as the first row specifying current step active nodes.
        :param obs: current observation. None observation means there is no observation available.
        :param cur_step_ids: current step node ids
        :param obs_node_ids: list of node ids related to the observation dimensions
        :return: a boolean showing whether an update is done, and the new observation array
        """
        # if timers is None:
        #     timers = [Timer(f'code {i}', level='debug') for i in range(10)]
        if not isinstance(cur_step_ids, set):
            cur_step_ids = set(cur_step_ids)
        # with timers[4]:
        new_active_indexes = [i for i in range(len(obs_node_ids)) if obs_node_ids[i] in cur_step_ids]
        dim = len(obs_node_ids)
        # with timers[5]:
        first_row = np.zeros(dim, dtype=bool)
        if obs is None and not new_active_indexes:
            return None
        else:
            # with timers[6]:
            first_row[new_active_indexes] = 1
            # with timers[7]:
            new_obs = np.vstack((first_row, obs)) if obs is not None else first_row.reshape((1, dim))
            return new_obs

    def _more_args(self, graph=None):
        return {}

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
    def extract_evidences(cls, trees, graph, **kwargs):
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


def train_models(cls, evidences, iterations, graph, save_in_db=False, project=None, **kwargs):
    try:
        return cls.train_models(evidences, iterations, graph, save_in_db, project, **kwargs)
    except:
        logger.error(traceback.format_exc())
        raise


def extract_evidences(cls, train_set, graph, trees, **kwargs):
    try:
        return cls.extract_evidences(trees, graph, **kwargs)
    except:
        logger.error(traceback.format_exc())
        raise


class NodeSeqLabelModel(SeqLabelDifModel, abc.ABC):
    """
    Node-based sequence labeling model
    """

    @classmethod
    def extract_evidences(cls, trees, graph, **kwargs):
        try:
            ''' evidences: dictionary of user id's to dictionaries in the format 
                            {'dimension': dimension, 'sequences': list_of_sequences} '''
            evidences = {}
            cascade_num = 1

            # Iterate each activation sequence and extract sequences of (observation, state) for each user
            for tree in trees:
                cascade_seqs = {}  # dictionary of user id's to the sequences of ObsPair instances for the current cascade
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
                    # logger.debugv('all_children = %s', all_children)

                    # Update the observation of each child and add the new observation-state to the current sequences.
                    for child_id in all_children:
                        if child_id not in activated and child_id in graph:
                            obs = observations.get(child_id)
                            parents = list(graph.predecessors(child_id))
                            new_obs = cls._update_obs(obs, cur_step_ids, parents)
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

                # Add current sequence of pairs (observation, state) to the seq labeling model evidences.
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

            logger.info('evidences extracted from %d cascades', len(trees))
            return evidences
        except:
            logger.error(traceback.format_exc())
            raise

    def _predict_one_thr(self, initial_tree, thr, graph, max_step=None):
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
                model = self._models.get(child_id)

                if model is not None:
                    logger.debugv('testing reshare to %s ...', child_id)
                    parents = list(graph.predecessors(child_id))

                    if child_id not in active_ids:
                        new_active_ids = cur_step & active_ids
                        obs = observations.get(child_id)
                        obs = self._update_obs(obs, new_active_ids, parents)

                        if obs is not None:
                            observations[child_id] = obs
                            logger.debugv('obs = \n%s', obs_to_str(obs))
                            node_id, _ = self._predict_by_obs(obs, thr, model, tree, parents)
                            if node_id:
                                tree.add_node(child_id, parent_id=node_id)
                                active_ids.add(child_id)
                                next_step.add(child_id)

                    else:
                        logger.debugv('user %s is already activated', child_id)

                else:
                    logger.debugv('user %s does not have any model', child_id)

                j += 1
                if j % 100 == 0:
                    logger.debugv('%d / %d of children iterated', j, len(all_children))
                    # for timer in timers:
                    #     if timer.sum:
                    #         timer.report_sum()

            cur_step = next_step
            step_num += 1

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
        # logger.debugv('initial observations:\n%s', pprint.pformat(obs_dic))
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
                model = self._models.get(child_id)

                if model is not None:
                    logger.debugv('testing reshare to %s ...', child_id)
                    parents = list(graph.predecessors(child_id))
                    activated = False
                    last_pred = None

                    for thr in threshold:
                        thr_active_ids = active_ids[thr]
                        if child_id not in thr_active_ids:
                            # with timers[1]:
                            new_active_ids = cur_step & thr_active_ids
                            obs = observations[thr].get(child_id)
                            # with timers[2]:
                            obs = self._update_obs(obs, new_active_ids, parents)

                            if obs is not None:
                                observations[thr][child_id] = obs
                                # if last_pred is None or not np.array_equal(obs, last_pred.obs):
                                #     logger.debugv('obs = \n%s', obs_to_str(obs))
                                logger.debugv('threshold %f ...', thr)
                                # with timers[3]:
                                node_id, pred = self._predict_by_obs(obs, thr, model, trees[thr], parents, last_pred)
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
                    logger.debugv('user %s does not have any model', child_id)

                j += 1
                if j % 100 == 0:
                    logger.debugv('%d / %d of children iterated', j, len(all_children))
                    # for timer in timers:
                    #     if timer.sum:
                    #         timer.report_sum()

            cur_step = next_step
            step_num += 1

        return trees

    def _get_predicted_node_id(self, obs, model, tree, obs_node_ids):
        # Set the parent with the maximum value of Lambda which is also activated at the current step as the
        # predicted parent of this child.
        conv_indexes = [model.orig_indexes_map.get(ind) for ind in np.where(obs[0, :])[0]]
        conv_indexes = list(filter(lambda x: x is not None, conv_indexes))
        if conv_indexes:
            max_lambda_ind = np.argmax(model.Lambda[conv_indexes])
            node_id = obs_node_ids[model.orig_indexes[conv_indexes[max_lambda_ind]]]
            if tree.get_node(node_id):
                return node_id
            else:
                logger.warning('parent node %s does not exist', node_id)
        else:
            logger.debugv('the newly active nodes are not available in the training data')
        return

    def _predict_by_obs(self, obs, thr, model, tree, obs_node_ids, last_pred=None):
        """

        :param obs: observation
        :param thr: threshold
        :param model: seq labeling model
        :param tree: current predicted tree
        :param obs_node_ids: list of node ids related to observation dimensions.
        :return: The parent id is returned if the diffusion is predicted, otherwise None.
        """
        if last_pred is not None and np.array_equal(obs, last_pred.obs):
            prob = last_pred.prob
        else:
            prob = model.get_prob(obs, self.active_state(), self.get_states())
            logger.debugv('prob = %f', prob)
            if prob == np.nan:
                logger.warning('activation prob. of obs. %s is nan', obs)
        pred = Prediction(obs, prob)

        if prob >= thr:
            node_id = self._get_predicted_node_id(obs, model, tree, obs_node_ids)
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
                    model = self._models.get(child_id)
                    if model is not None:
                        # Update the observation of this child.
                        parents = parents_dic[child_id]
                        obs = observations.get(child_id)
                        obs = self._update_obs(obs, cur_step_ids, parents)
                        observations[child_id] = obs
                        # logger.debugv('obs set: %s', observations[child_id])

            next_step = reduce(lambda x, y: x + y, [node.children for node in cur_step], [])
            cur_step = next_step
            i += 1

        return observations

    def _get_evid_manager(self, project):
        return EvidenceManager(project)


class Prediction:
    def __init__(self, obs, prob, state=True):
        self.obs = obs
        self.prob = prob
        self.state = state


class MultiStateModel(NodeSeqLabelModel, abc.ABC):
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

    def _get_evid_manager(self, project):
        return ParentSensEvidManager(project)

    def _predict_by_obs(self, obs, thr, model, tree, obs_node_ids, last_pred=None):
        if last_pred is not None and np.array_equal(obs, last_pred.obs):
            state, prob = last_pred.state, last_pred.prob
        else:
            all_states = list(range(len(obs_node_ids) + 1))
            probs = model.get_probs(obs, all_states)
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
