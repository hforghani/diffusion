import pprint

import settings
from log_levels import DEBUG_LEVELV_NUM
from db.managers import SeqLabelDBManager, EdgeEvidenceManager, EdgeMEMMManager, ParentSensEvidManager
from diffusion.enum import Method
from seq_labeling.pgm import *
from seq_labeling.models import SeqLabelDifModel, NodeSeqLabelModel, Prediction
from seq_labeling.utils import obs_to_str
from settings import logger


class MEMMModel(SeqLabelDifModel, abc.ABC):
    max_iterations = 500

    @classmethod
    def _get_seq_label_manager(cls, project):
        return SeqLabelDBManager(project, cls.method)

    @classmethod
    def _train_model(cls, evidence, iterations, key, graph, **kwargs):
        states = cls.get_states(key, graph)
        if 'td_param' in kwargs:
            logger.debug('td_param = %s', kwargs['td_param'])
        memm = cls.get_memm_instance(kwargs.get('td_param'))
        logger.debug('type(memm) = %s', type(memm))
        memm.fit(evidence, iterations, states)
        return memm

    @staticmethod
    @abc.abstractmethod
    def get_memm_instance(td_param):
        pass


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
            obs_nodes_ids_map = kwargs['obs_nodes_ids_map']
        except KeyError:
            raise ValueError('Keyword argument obs_nodes_ids_map must be given')
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
                    obs_node_is = obs_nodes_ids_map[edge]
                    obs = observations.get(edge)
                    new_obs = cls._update_obs(obs, cur_step_ids, obs_node_is)
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
                        dim = len(obs_nodes_ids_map[edge])
                    evidences.setdefault(edge, {
                        'dimension': dim,
                        'sequences': []
                    })
                    evidences[edge]['sequences'].append(cascade_seqs[edge])

                    logger.info('evidences extracted from %d cascades', len(train_set))
        return evidences

    def predict(self, initial_tree, graph, thresholds, max_step=None, obs_node_ids_map=None):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :return: dictionary of thresholds to their predicted trees
        """
        if obs_node_ids_map is None:
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

        obs_dic = self._get_initial_observations(initial_tree, cur_step_nodes, graph, obs_node_ids_map)
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
                    memm = self.get_model(edge)

                    if memm is not None:
                        logger.debugv('testing reshare from %s to %s ...', node_id, child_id)

                        activated = False
                        last_prob, last_obs = None, None

                        for thr in thresholds:
                            thr_active_ids = active_ids[thr]
                            if child_id not in thr_active_ids:
                                obs_node_ids = obs_node_ids_map[edge]
                                new_active_ids = cur_step & thr_active_ids
                                obs = observations[thr].get(edge)
                                obs = self._update_obs(obs, new_active_ids, obs_node_ids)

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

    def _get_initial_observations(self, initial_tree, max_depth_nodes, graph, obs_node_ids_map):
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
                obs_node_ids = obs_node_ids_map[edge]
                obs = observations[edge]
                new_obs = self._update_obs(obs, cur_step_ids, obs_node_ids)
                observations[edge] = new_obs

            next_step = reduce(lambda x, y: x + y, [node.children for node in cur_step], [])
            cur_step = next_step
            i += 1

        if settings.LOG_LEVEL <= DEBUG_LEVELV_NUM:
            logger.debugv('initial observations =\n%s', pprint.pformat(observations))
        return observations

    def _more_args(self, graph):
        return dict(obs_node_ids_map=self.extract_obs_node_ids_map(graph))

    @staticmethod
    def extract_obs_node_ids_map(graph):
        logger.info('extracting edge dimension users ...')
        obs_nodes_ids_map = {}
        for edge in graph.edges():
            src_children = graph.successors(edge[0])
            dest_parents = graph.predecessors(edge[1])
            obs_nodes_ids_map[edge] = sorted(set(src_children) | set(dest_parents) - {edge[1]})
        return obs_nodes_ids_map

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
    def _get_seq_label_manager(cls, project):
        return EdgeMEMMManager(project, cls.method)


class NodeMEMMModel(NodeSeqLabelModel, MEMMModel, abc.ABC):
    pass


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


class LongMEMMModel(NodeMEMMModel):
    method = Method.LONG_MEMM

    # max_iterations = 1000

    @staticmethod
    def get_memm_instance(td_param=None):
        return LongMEMM(td_param) if td_param is not None else LongMEMM()

    def _get_predicted_node_id(self, obs, model, tree, obs_node_ids):
        # Set the parent with the maximum value of Lambda which is also activated at the current step as the
        # predicted parent of this child.
        obs_dim = len(obs_node_ids)
        conv_indexes = [model.orig_indexes_map.get(ind) for ind in np.where(obs[0, :obs_dim])[0]]
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
