import pprint
import traceback
from functools import reduce
from random import shuffle
import numpy as np
from bson import ObjectId
from networkx import DiGraph

from db.managers import MEMMManager
from cascade.enum import Method
from memm.memm import BinMEMM, TDMEMM, ParentTDMEMM, LongParentTDMEMM
from memm.exceptions import MemmException
from settings import logger


def train_memms(evidences, method, save_in_db=False, project=None):
    try:
        user_ids = list(evidences.keys())
        shuffle(user_ids)
        logger.debugv('training memms started')
        memms = {}
        count = 0
        manager = MEMMManager(project, method) if save_in_db else None
        graph = None
        if method in [Method.PARENT_SENS_TD_MEMM, Method.LONG_PARENT_SENS_TD_MEMM]:
            graph = project.load_or_extract_graph()

        for uid in user_ids:
            count += 1
            ev = evidences.pop(uid)  # to free RAM
            logger.debug('training MEMM %d (user id: %s, dimensions: %d) ...', count, uid, ev['dimension'])

            states = [False, True]
            if method in [Method.BIN_MEMM, Method.REDUCED_BIN_MEMM]:
                memm = BinMEMM()
            elif method in [Method.PARENT_SENS_TD_MEMM, Method.LONG_PARENT_SENS_TD_MEMM]:
                states = list(range(graph.in_degree(uid) + 1))
                if method == Method.PARENT_SENS_TD_MEMM:
                    memm = ParentTDMEMM()
                else:
                    memm = LongParentTDMEMM()
            else:
                memm = TDMEMM()

            try:
                memm.fit(ev, states)
                memms[uid] = memm
            except MemmException:
                logger.warn('evidences for user %s ignored due to insufficient data', uid)
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
    except:
        logger.error(traceback.format_exc())
        raise


def extract_bin_memm_evidences(train_set, graph, act_seqs):
    try:
        evidences = {}  # dictionary of user id's to list of the sequences of ObsPair instances.
        parent_sizes = {}  # dictionary of user id's to number of their parents
        count = 0

        # Iterate each activation sequence and extract sequences of (observation, state) for each user
        for cascade_id in train_set:
            cascade_seqs = {}  # dictionary of user id's to the sequences of ObsPair instances for the current cascade
            act_seq = act_seqs[cascade_id]
            observations = {}  # current observation of each user
            activated = set()  # set of current activated users
            i = 0
            logger.info('cascade %d with %d users ...', count + 1, len(act_seq.users))

            for uid in act_seq.users:  # Notice users are sorted by activation time.
                activated.add(uid)
                parents_count = graph.in_degree(uid) if uid in graph else 0
                parent_sizes[uid] = parents_count
                children = list(graph.successors(uid)) if uid in graph else []

                # Put the last observation with state 1 in the sequence of (observation, state) if exists.
                if parents_count and uid in cascade_seqs:
                    observations.setdefault(uid, np.zeros(parents_count, dtype=bool))  # initial observation: 0000000
                    uid_cur_seqs = cascade_seqs[uid]
                    if uid_cur_seqs:
                        obs = uid_cur_seqs[-1][0]
                        del uid_cur_seqs[-1]
                        uid_cur_seqs.append((obs, True))

                if children:
                    logger.debug('iterating on %d children ...', len(children))

                # Set the related coefficient of the children observations equal to 1.
                j = 0
                for child in children:
                    if child not in activated:
                        child_parents = list(graph.predecessors(child)) if child in graph else []
                        if child_parents:
                            parent_sizes[child] = len(child_parents)
                            obs = observations.setdefault(child, np.zeros(len(child_parents),
                                                                          dtype=bool))  # initial observation: 0000000
                            index = child_parents.index(uid)
                            obs[index] = True
                            cascade_seqs.setdefault(child, [(np.zeros(len(child_parents), dtype=bool), False)])
                            cascade_seqs[child].append((obs.copy(), False))
                    j += 1
                    if j % 1000 == 0:
                        logger.debug('%d children done', j)

                i += 1
                if i % 100 == 0:
                    logger.debug('%d users done', i)

            count += 1
            if count % 1000 == 0:
                logger.info('%d cascades done', count)

            # Add current sequence of pairs (observation, state) to the MEMM evidences.
            logger.debug('adding sequences of current cascade ...')
            for uid in cascade_seqs:
                dim = parent_sizes[uid]
                evidences.setdefault(uid, {
                    'dimension': dim,
                    'sequences': []
                })
                evidences[uid]['sequences'].append(cascade_seqs[uid])

        return evidences
    except:
        logger.error(traceback.format_exc())
        raise


def extract_reduced_memm_evidences(train_set, graph, trees, method):
    try:
        evidences = {}  # dictionary of user id's to list of the sequences of ObsPair instances.
        cascade_num = 1

        # Iterate each activation sequence and extract sequences of (observation, state) for each user
        for cascade_id in train_set:
            cascade_seqs = {}  # dictionary of user id's to the sequences of ObsPair instances for the current cascade
            tree = trees[cascade_id]
            observations = {}  # current observation of each user
            activated = set()  # set of current activated users
            logger.info('cascade %d ...', cascade_num)

            cur_step = tree.roots
            step_num = 1

            while cur_step:
                logger.debug('step %d with %d users started', step_num, len(cur_step))
                logger.debug('extracting parents and children ...')
                cur_step_ids = {node.user_id for node in cur_step}

                # Put the state of the last observation in the sequence of (observation, state) equal to 1 (activated)
                # for all nodes in the current step.
                for node in cur_step:
                    uid = node.user_id
                    activated.add(uid)
                    parents_count = graph.in_degree(uid) if uid in graph else 0

                    if parents_count and uid in cascade_seqs and uid in observations:
                        if cascade_seqs[uid]:
                            obs = cascade_seqs[uid][-1][0]
                            state = active_state(method, node, graph)
                            cascade_seqs[uid] = cascade_seqs[uid][:-1] + [(obs, state)]
                            logger.debugv('(obs, state) updated for %s : (%s, %d)', uid, obs, state)
                        del observations[uid]

                # Get the children whom at least one of their parents are in the current step.
                children_sets = (set(graph.successors(node_id)) for node_id in cur_step_ids if node_id in graph)
                all_children = list(reduce(lambda x, y: x | y, children_sets, set()))
                logger.debugv('all_children = %s', all_children)

                # Update the observation of each child and add the new observation-state to the current sequences.
                for child_id in all_children:
                    if child_id not in activated:
                        if child_id not in graph:
                            continue
                        zero_obs = get_zero_obs(graph.in_degree(child_id), method)  # initial observation: 0000000
                        obs = observations.setdefault(child_id, zero_obs.copy())
                        new_obs = get_new_obs(child_id, cur_step_ids, obs, graph)
                        observations[child_id] = new_obs
                        state = inactive_state(method)
                        cascade_seqs.setdefault(child_id, [(zero_obs.copy(), state)])
                        cascade_seqs[child_id].append((new_obs, state))
                        logger.debugv('(obs, state) added for %s : (%s, %d)', child_id, new_obs, state)

                # Update the current step nodes.
                cur_step = reduce(lambda li1, li2: li1 + li2, [node.children for node in cur_step])

                # Update the observations for the next step if necessary.
                if cur_step:
                    set_next_state_observations(observations, method)

                logger.debug('%d steps done', step_num)
                step_num += 1

            if cascade_num % 1000 == 0:
                logger.info('%d cascades done', cascade_num)
            cascade_num += 1

            # Add current sequence of pairs (observation, state) to the MEMM evidences.
            logger.debug('adding sequences of current cascade ...')
            logger.debugv('cascade_seqs =\n%s', pprint.pformat(cascade_seqs))
            for uid in cascade_seqs:
                dim = graph.in_degree(uid)
                evidences.setdefault(uid, {
                    'dimension': dim,
                    'sequences': []
                })
                evidences[uid]['sequences'].append(cascade_seqs[uid])

        return evidences
    except:
        logger.error(traceback.format_exc())
        raise


def get_zero_obs(dim, method):
    if method in [Method.BIN_MEMM, Method.REDUCED_BIN_MEMM]:
        return np.zeros(dim, dtype=bool)
    else:
        return np.zeros(dim)


def set_next_state_observations(observations, method):
    if method in [Method.TD_MEMM,
                  Method.REDUCED_TD_MEMM,
                  Method.PARENT_SENS_TD_MEMM,
                  Method.LONG_PARENT_SENS_TD_MEMM]:
        divide_obs_by_2(observations)


def divide_obs_by_2(observations):
    for user_id, obs in observations.items():
        observations[user_id] = obs / 2


def get_new_obs(child_id: ObjectId, cur_step_ids: set, obs: np.ndarray, graph: DiGraph) -> np.ndarray:
    parents = list(graph.predecessors(child_id))
    cur_step_parents = set(parents) & cur_step_ids
    parent_indexes = [parents.index(uid) for uid in cur_step_parents]
    new_obs = obs.copy()
    new_obs[parent_indexes] = 1

    return new_obs


def inactive_state(method):
    if method in [Method.PARENT_SENS_TD_MEMM, Method.LONG_PARENT_SENS_TD_MEMM]:
        return 0
    else:
        return False


def active_state(method, node, graph):
    if method in [Method.PARENT_SENS_TD_MEMM, Method.LONG_PARENT_SENS_TD_MEMM]:
        parents = list(graph.predecessors(node.user_id))
        index = parents.index(node.parent_id)
        return index + 1
    else:
        return True
