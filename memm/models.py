import json
import math
import multiprocessing
import os
import random
import time
import traceback
from multiprocessing.pool import Pool

from bson.objectid import ObjectId
from pympler.asizeof import asizeof

from cascade.models import CascadeNode, CascadeTree, ParamTypes
from memm.asyncronizables import train_memms, test_memms, test_memms_eco
from memm.db import MEMMManager
from memm.memm import MEMM
# from neo4j.models import Neo4jGraph
import numpy as np
from settings import logger, BASEPATH, mongodb
from utils.time_utils import Timer

MEMM_EVID_FILE_NAME = 'memm/evidence'


# MEMM_EVID_FILE_NAME = 'memm/evidence-5d88f41e86887707d4526076'


class MEMMModel:
    def __init__(self, project):
        self.project = project
        self.__memms = {}

    def __save_evidences(self, evidences):
        logger.info('saving MEMM evidences ...')
        evidences = {str(key): value for key, value in evidences.items()}
        self.project.save_param(evidences, MEMM_EVID_FILE_NAME, ParamTypes.JSON)
        logger.info('done')

    def __load_evidences(self):
        logger.info('loading MEMM evidences ...')
        evidences = self.project.load_param(MEMM_EVID_FILE_NAME, ParamTypes.JSON)
        evidences = {ObjectId(uid): evidences[uid] for uid in evidences}
        return evidences

    def __add_and_save_evidences(self, new_evidences):
        try:
            evidences = self.__load_evidences()
        except FileNotFoundError:
            evidences = {}

        if new_evidences:
            logger.info('integrating MEMM evidences ...')
            for uid in new_evidences:
                if uid not in evidences:
                    evidences[uid] = new_evidences[uid]
                else:
                    evidences[uid][1].extend(new_evidences[uid][1])
            self.__save_evidences(evidences)

        return evidences

    def __prepare_evidences(self, train_set):
        """
        Prepare the sequence of observations and states to train the MEMM models.
        :param train_set: list of training cascade id's
        :return: a dictionary of user id's to instances of MemmEvidence
        """
        act_seqs = self.project.load_or_extract_act_seq()

        try:
            evidences = self.__load_evidences()

        except FileNotFoundError:
            logger.info('no evidences found! extraction started')
            count = 0
            new_evidences = {}  # dictionary of user id's to list of the sequences of ObsPair instances which are not saved yet.
            cascade_seqs = {}  # dictionary of user id's to the sequences of ObsPair instances for this current cascade
            parent_sizes = {}  # dictionary of user id's to number of their parents

            logger.info('extracting sequences from %d cascades ...', len(train_set))

            # Iterate each activation sequence and extract sequences of (observation, state) for each user
            for cascade_id in train_set:
                act_seq = act_seqs[cascade_id]
                observations = {}  # current observation of each user
                activated = set()  # set of current activated users
                i = 0
                logger.info('cascade %d with %d users ...', count + 1, len(act_seq.users))

                # relations = mongodb.relations.find({'user_id': {'$in': act_seq.users}})
                # rel_dic = {rel['user_id']: rel for rel in relations}

                for uid in act_seq.users:  # Notice users are sorted by activation time.
                    activated.add(uid)
                    # parents_count = len(rel_dic[uid]['parents'])
                    rel = mongodb.relations.find_one({'user_id': uid}, {'_id': 0, 'children': 1, 'parents': 1})
                    parents_count = len(rel['parents']) if rel is not None else 0
                    parent_sizes[uid] = parents_count
                    logger.debug('extracting children ...')
                    # children = rel_dic[uid]['children']
                    children = rel['children'] if rel is not None else []

                    # Put the last observation with state 1 in the sequence of (observation, state).
                    if parents_count:
                        observations.setdefault(uid, 0)  # initial observation: 0000000
                        cascade_seqs.setdefault(uid, [])
                        if cascade_seqs[uid]:
                            obs = cascade_seqs[uid][-1][0]
                            del cascade_seqs[uid][-1]
                            cascade_seqs[uid].append((obs, 1))

                    if children:
                        logger.debug('iterating on %d children ...', len(children))

                    # relations = mongodb.relations.find({'user_id': {'$in': cur_step}}, {'_id':0, 'user_id':1, 'parents':1})
                    # parents_dic = {rel['user_id']: rel['parents'] for rel in relations}

                    # Set the related coefficient of the children observations equal to 1.
                    j = 0
                    for child in children:
                        # if child not in parents_dic:
                        #     continue
                        # child_parents = parents_dic[child]
                        rel = mongodb.relations.find_one({'user_id': child}, {'_id': 0, 'parents': 1})
                        if rel is None:
                            continue
                        child_parents = rel['parents']
                        parent_sizes[child] = len(child_parents)

                        obs = observations.setdefault(child, 0)
                        index = child_parents.index(uid)
                        obs |= 1 << (len(child_parents) - index - 1)
                        observations[child] = obs
                        if child not in activated:
                            cascade_seqs.setdefault(child, [(0, 0)])
                            cascade_seqs[child].append((obs, 0))
                        j += 1
                        if j % 1000 == 0:
                            logger.debug('%d children done', j)

                    i += 1
                    logger.debug('%d users done', i)

                count += 1
                if count % 1000 == 0:
                    logger.info('%d cascades done', count)

                # Add current sequence of pairs (observation, state) to the MEMM evidences.
                logger.info('adding sequences of current cascade ...')
                for uid in cascade_seqs:
                    # dim = len(rel_dic[uid]['parents'])
                    dim = parent_sizes[uid]
                    new_evidences.setdefault(uid, [dim, []])
                    new_evidences[uid][1].append(cascade_seqs[uid])
                cascade_seqs = {}

                if len(act_seq.users) > 1000:
                    self.__add_and_save_evidences(new_evidences)
                    new_evidences = {}

            evidences = self.__add_and_save_evidences(new_evidences)

        return evidences

    def __separate_big_ev(self, evidences):
        big_userids = []
        notbig_userids = []
        big_threshold = 60000
        for uid in evidences:
            if asizeof(evidences[uid][1]) > big_threshold:
                big_userids.append(uid)
            else:
                notbig_userids.append(uid)
        # Shuffle not-big user ids' list to balance the process memory sizes.
        random.shuffle(notbig_userids)
        return big_userids, notbig_userids

    def __fit_multiproc(self, evidences):
        """
        Side effect: Clears the evidences dictionary.
        """
        user_ids = list(evidences.keys())
        random.shuffle(user_ids)
        # process_count = multiprocessing.cpu_count()
        process_count = min(8, multiprocessing.cpu_count())
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
            res = pool.apply_async(train_memms, (evidences_i,))
            results.append(res)

        del evidences  # to free RAM
        pool.close()
        pool.join()

        # Collect results of the processes.
        logger.debug('assembling multiprocessed results of MEMM training ...')
        for res in results:
            memms_i = res.get()
            user_ids_i = list(memms_i.keys())
            for uid in user_ids_i:
                self.__memms[uid] = memms_i.pop(uid)  # to free RAM
        logger.debug('assembling done')

    def __fit_by_evidences(self, train_set, evidences):
        # Divide evidences into some parts. Each time load a part from evidences and train the
        # corresponding MEMMS to avoid high RAM consumption.
        logger.info('separating big evidences ...')
        big_user_ids, notbig_user_ids = self.__separate_big_ev(evidences)
        logger.debug('%d big and %d not-big evidences considered', len(big_user_ids), len(notbig_user_ids))
        big_ev = {}
        notbig_ev = {}
        for uid in big_user_ids:
            big_ev[uid] = evidences.pop(uid)  # to free RAM
        for uid in notbig_user_ids:
            notbig_ev[uid] = evidences.pop(uid)  # to free RAM

        # Train not-big evidences multi-processing in multi-processing mode.

        parts_count = 5
        part_size = int(math.ceil(len(notbig_user_ids) / parts_count))

        for j in range(parts_count):
            logger.info('loading evidences of part %d of users', j + 1)
            part_j = notbig_user_ids[j * part_size: (j + 1) * part_size]
            evidences_j = {}
            for uid in part_j:
                evidences_j[uid] = notbig_ev.pop(uid)  # to free RAM
            logger.info("training %d MEMM's related to part %d of users ...", len(part_j), j + 1)
            self.__fit_multiproc(evidences_j)

        # Train big evidences sequentially.
        logger.info('training %d big MEMMs sequentially', len(big_ev))
        memms = train_memms(big_ev)
        del big_ev
        for uid in big_user_ids:
            self.__memms[uid] = memms.pop(uid)  # to free RAM

        logger.info('training MEMMs finished')

    def fit(self, train_set):
        """
        Train MEMM's for each user in training set.
        :param train_set:   cascade id's in training set
        :return:            self
        """
        self.__load_memms()
        remaining_uids = set(train_set) - set(self.__memms.keys())
        logger.info('%d MEMMS not trained. going to train ...', len(remaining_uids))
        if remaining_uids:
            evidences = self.__prepare_evidences(train_set)
            evidences = {uid: value for uid, value in evidences.items() if uid in remaining_uids}
            self.__fit_by_evidences(train_set, evidences)
        self.__save_memms(remaining_uids)

        return self

    def __predict_multiproc(self, children, parent_node, parents_dic, observations, active_ids, threshold, next_step):
        children_copy = children.copy()
        random.shuffle(children_copy)

        process_count = multiprocessing.cpu_count() - 1
        # process_count = 4
        logger.debug('starting %d processes to predict by MEMMs', process_count)
        pool = Pool(processes=process_count)
        step = int(math.ceil(len(children) / process_count))
        results = []

        for i in range(process_count):
            children_i = children[i * step: (i + 1) * step]
            parents_dic_i = {}
            for uid in children_i:
                parents_dic_i[uid] = parents_dic.pop(uid)  # to free RAM
            observations_i = {uid: observations[uid] for uid in children_i}
            memms_i = {uid: self.__memms[uid] for uid in children_i if uid in self.__memms}

            # Train a MEMM for each user.
            res = pool.apply_async(test_memms,
                                   (children_i, parents_dic_i, observations_i, active_ids, memms_i, threshold))
            results.append(res)

        del parents_dic  # to free RAM
        pool.close()
        pool.join()

        # Collect results of the processes.
        logger.debug('assembling multi-processed results of MEMM predictions ...')
        for res in results:
            act_children = res.get()
            for child_id in act_children:
                child = CascadeNode(child_id)
                parent_node.children.append(child)
                next_step.append(child)
                active_ids.append(child_id)

        logger.debug('assembling done')

    def __predict_multiproc_eco(self, children, parent_node, parents_dic, observations, active_ids, threshold,
                                next_step):
        children_copy = children.copy()
        random.shuffle(children_copy)

        process_count = multiprocessing.cpu_count() - 1
        # process_count = 4
        logger.debug('starting %d processes to predict by MEMMs', process_count)
        pool = Pool(processes=process_count)
        step = int(math.ceil(len(children) / process_count))
        results = []

        for i in range(process_count):
            children_i = children[i * step: (i + 1) * step]
            parents_dic_i = {}
            for uid in children_i:
                parents_dic_i[uid] = parents_dic.pop(uid)  # to free RAM
            observations_i = {uid: observations[uid] for uid in children_i}

            # Train a MEMM for each user.
            res = pool.apply_async(test_memms_eco,
                                   (children_i, parents_dic_i, observations_i, active_ids, threshold))
            results.append(res)

        del parents_dic  # to free RAM
        pool.close()
        pool.join()

        # Collect results of the processes.
        logger.debug('assembling multi-processed results of MEMM predictions ...')
        for res in results:
            act_children = res.get()
            for child_id in act_children:
                child = CascadeNode(child_id)
                parent_node.children.append(child)
                next_step.append(child)
                active_ids.append(child_id)

        logger.debug('assembling done')

    def predict(self, initial_tree, threshold=None, max_step=None):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :return:         Predicted tree
        """
        ch_timer = Timer('get children')
        p_timer = Timer('get parents')
        obs_timer = Timer('observations')
        m_timer = Timer('MEMM predict')

        if not isinstance(initial_tree, CascadeTree):
            raise ValueError('tree must be CascadeTree')
        tree = initial_tree.copy()

        # Find initially activated nodes.
        cur_step = sorted(tree.get_leaves(), key=lambda n: n.datetime)  # Set tree nodes as initial step.
        active_ids = initial_tree.node_ids()
        step_num = 1

        # Create dictionary of current observations of the nodes.
        observations = {uid: 0 for uid in active_ids}

        # Predict the cascade tree.
        # At each iteration find newly activated nodes based on MEMM probabilities and add them to the tree.
        while cur_step and (max_step is None or step_num <= max_step):
            logger.debug('predicting step %d ...', step_num)

            next_step = []

            with ch_timer:
                relations = mongodb.relations.find({'user_id': {'$in': [n.user_id for n in cur_step]}},
                                                   {'_id': 0, 'user_id': 1, 'children': 1})
                children_dic = {rel['user_id']: rel['children'] for rel in relations}

            i = 0
            for node in cur_step:
                node_id = node.user_id
                children = children_dic.pop(node_id, [])  # to get and free RAM
                # rel = mongodb.relations.find_one({'user_id': node_id}, {'_id': 0, 'children': 1})
                # children = rel['children'] if rel is not None else []

                if not children:
                    continue

                logger.debug('num of children of %s : %d', node_id, len(children))

                # Add all children if threshold is 0.
                if threshold == 0:
                    j = 0
                    inact_children = set(children) - set(active_ids)
                    for child_id in inact_children:
                        child = CascadeNode(child_id)
                        node.children.append(child)
                        next_step.append(child)
                        active_ids.append(child_id)
                        j += 1
                        if j % 100 == 0:
                            logger.debugv('%d / %d of children iterated', j, len(inact_children))

                elif threshold < 1:
                    with p_timer:
                        relations = mongodb.relations.find({'user_id': {'$in': children}},
                                                           {'_id': 0, 'user_id': 1, 'parents': 1})
                        parents_dic = {rel['user_id']: rel['parents'] for rel in relations}

                    with obs_timer:
                        for child_id in children:
                            obs = observations.setdefault(child_id, 0)
                            if child_id not in parents_dic:
                                continue
                            parents = parents_dic[child_id]
                            index = parents.index(node_id)
                            obs |= 1 << (len(parents) - index - 1)
                            observations[child_id] = obs

                    with m_timer:
                        if len(children) > 150000:
                            self.__predict_multiproc_eco(children, node, parents_dic, observations, active_ids,
                                                         threshold, next_step)
                        elif len(children) > 1000:
                            self.__predict_multiproc_eco(children, node, parents_dic, observations, active_ids,
                                                         threshold,
                                                         next_step)
                        else:
                            memms_i = {uid: self.__memms[uid] for uid in children if uid in self.__memms}
                            act_children = test_memms(children, parents_dic, observations, active_ids, memms_i,
                                                      threshold)
                            for child_id in act_children:
                                child = CascadeNode(child_id)
                                node.children.append(child)
                                next_step.append(child)
                                active_ids.append(child_id)

                i += 1
                logger.debug('%d / %d nodes of current step done', i, len(cur_step))

            cur_step = next_step
            step_num += 1

        for timer in [ch_timer, p_timer, obs_timer, m_timer]:
            timer.report_sum()

        return tree

    def __save_memms(self, user_ids):
        MEMMManager.insert_many(self.__memms, user_ids)

    def __load_memms(self):
        self.__memms = MEMMManager.fetch_all()
