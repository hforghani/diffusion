import pprint
import traceback
from functools import reduce
from random import shuffle
import numpy as np
from bson import ObjectId

from db.managers import MEMMManager
from memm.enum import MEMMMethod
from memm.memm import MEMM, BinMEMM, FloatMEMM
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

        for uid in user_ids:
            count += 1
            ev = evidences.pop(uid)  # to free RAM
            logger.debug('training MEMM %d (user id: %s, dimensions: %d) ...', count, uid, ev['dimension'])

            if method == MEMMMethod.BIN_MEMM:
                memm = BinMEMM()
            else:
                memm = FloatMEMM()

            try:
                memm.fit(ev)
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


def extract_float_memm_evidences(train_set, graph, trees):
    try:
        evidences = {}  # dictionary of user id's to list of the sequences of ObsPair instances.
        cascade_num = 1

        # Iterate each activation sequence and extract sequences of (observation, state) for each user
        for cascade_id in train_set:
            cascade_seqs = {}  # dictionary of user id's to the sequences of ObsPair instances for the current cascade
            tree = trees[cascade_id]
            observations = {}  # current observation of each user
            activated = set()  # set of current activated users
            parent_sizes = {}  # dictionary of user id's to number of their parents
            logger.info('cascade %d ...', cascade_num)

            cur_step = tree.roots
            step_num = 1

            while cur_step:
                logger.debug('step %d with %d users started', step_num, len(cur_step))
                logger.debug('extracting parents and children ...')
                for node in cur_step:
                    if node.user_id in graph:
                        parent_sizes[node.user_id] = graph.in_degree(node.user_id)

                # Put the state of the last observation in the sequence of (observation, state) equal to 1 (activated)
                # for all nodes in the current step.
                for node in cur_step:
                    uid = node.user_id
                    activated.add(uid)
                    parents_count = parent_sizes.get(uid, 0)

                    if parents_count and uid in cascade_seqs and uid in observations:
                        uid_cur_seqs = cascade_seqs[uid]
                        if uid_cur_seqs:
                            obs = uid_cur_seqs[-1][0]
                            del uid_cur_seqs[-1]
                            uid_cur_seqs.append((obs, True))
                        del observations[uid]

                user_num = 1
                children_to_add = set()

                for node in cur_step:
                    uid = node.user_id
                    children = list(graph.successors(uid)) if uid in graph else []

                    if children:
                        logger.debug('iterating on %d children ...', len(children))

                    # Set the related coefficient of the observations of all children equal to 1.
                    child_num = 1
                    for child in children:
                        if child not in activated:
                            child_parents = list(graph.predecessors(child)) if child in graph else []
                            if child_parents:
                                parent_sizes[child] = len(child_parents)
                                obs = observations.setdefault(child, np.zeros(
                                    len(child_parents)))  # initial observation: 0000000
                                index = child_parents.index(uid)
                                obs[index] = 1
                                # child_node = tree.get_node(child)
                                # obs[index] = 1 if child_node and uid == child_node.parent_id else 0.5
                                children_to_add.add(child)
                        if child_num % 1000 == 0:
                            logger.debugv('%d children done', child_num)
                        child_num += 1

                    logger.debug('cascade %d -> step %d -> %d users done', cascade_num, step_num, user_num)
                    user_num += 1

                for child in children_to_add:
                    obs = observations[child]
                    cascade_seqs.setdefault(child, [(np.zeros(obs.shape[0]), False)])
                    cascade_seqs[child].append((obs.copy(), False))

                # Update current step nodes.
                cur_step = reduce(lambda li1, li2: li1 + li2, [node.children for node in cur_step])

                # Divide all observations by 2 at the end of the step to apply the latency impact on activation.
                if cur_step:
                    for uid, obs in observations.items():
                        observations[uid] = obs / 2

                logger.debug('%d steps done', step_num)
                step_num += 1

            if cascade_num % 1000 == 0:
                logger.info('%d cascades done', cascade_num)
            cascade_num += 1

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


def extract_parent_sens_float_memm_evidences(train_set, graph, trees):
    try:
        evidences = {}  # dictionary of user id's to list of the sequences of ObsPair instances.
        cascade_num = 1

        # Iterate each activation sequence and extract sequences of (observation, state) for each user
        for cascade_id in train_set:
            cascade_seqs = {}  # dictionary of user id's to the sequences of ObsPair instances for the current cascade
            tree = trees[cascade_id]
            observations = {}  # current observation of each user
            activated = set()  # set of current activated users
            parent_sizes = {}  # dictionary of user id's to number of their parents
            logger.info('cascade %d ...', cascade_num)

            cur_step = tree.roots
            step_num = 1

            while cur_step:
                logger.debug('step %d with %d users started', step_num, len(cur_step))
                logger.debug('extracting parents and children ...')
                for node in cur_step:
                    if node.user_id in graph:
                        parent_sizes[node.user_id] = graph.in_degree(node.user_id)

                # Put the state of the last observation in the sequence of (observation, state) equal to 1 (activated)
                # for all nodes in the current step.
                for node in cur_step:
                    uid = node.user_id
                    activated.add(uid)
                    parents_count = parent_sizes.get(uid, 0)

                    if parents_count and uid in cascade_seqs and uid in observations:
                        uid_cur_seqs = cascade_seqs[uid]
                        if uid_cur_seqs:
                            obs = uid_cur_seqs[-1][0]
                            del uid_cur_seqs[-1]
                            uid_cur_seqs.append((obs, True))
                        del observations[uid]

                user_num = 1
                children_to_add = set()

                for node in cur_step:
                    uid = node.user_id
                    children = list(graph.successors(uid)) if uid in graph else []

                    if children:
                        logger.debug('iterating on %d children ...', len(children))

                    # Set the related coefficient of the observations of all children equal to 1.
                    child_num = 1
                    for child in children:
                        if child not in activated:
                            child_parents = list(graph.predecessors(child)) if child in graph else []
                            if child_parents:
                                parent_sizes[child] = len(child_parents)
                                obs = observations.setdefault(child, np.zeros(
                                    len(child_parents)))  # initial observation: 0000000
                                index = child_parents.index(uid)

                                # Different with Float MEMM:
                                child_node = tree.get_node(child)
                                obs[index] = 1 if child_node and uid == child_node.parent_id else 0.5

                                children_to_add.add(child)
                        if child_num % 1000 == 0:
                            logger.debugv('%d children done', child_num)
                        child_num += 1

                    logger.debug('cascade %d -> step %d -> %d users done', cascade_num, step_num, user_num)
                    user_num += 1

                for child in children_to_add:
                    obs = observations[child]
                    cascade_seqs.setdefault(child, [(np.zeros(obs.shape[0]), False)])
                    cascade_seqs[child].append((obs.copy(), False))

                # Update current step nodes.
                cur_step = reduce(lambda li1, li2: li1 + li2, [node.children for node in cur_step])

                # Divide all observations by 2 at the end of the step to apply the latency impact on activation.
                if cur_step:
                    for uid, obs in observations.items():
                        observations[uid] = obs / 2

                logger.debug('%d steps done', step_num)
                step_num += 1

            if cascade_num % 1000 == 0:
                logger.info('%d cascades done', cascade_num)
            cascade_num += 1

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
