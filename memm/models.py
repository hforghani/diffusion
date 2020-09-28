import json
import math
import multiprocessing
import os
import time
from multiprocessing.pool import Pool

from bson import ObjectId
from pympler.asizeof import asizeof

from cascade.models import CascadeNode, CascadeTree, ParamTypes
from memm.memm import MEMM, MemmException, times
# from neo4j.models import Neo4jGraph
import numpy as np
from settings import logger, BASEPATH, mongodb

MEMM_EVID_FILE_NAME = 'memm/evidence'


# MEMM_EVID_FILE_NAME = 'memm/evidence-5d88f41e86887707d4526076'


def train_memms(evidences):
    user_ids = list(evidences.keys())
    p_name = multiprocessing.current_process().name
    logger.debugv('[%s] training memms started', p_name)
    memms = {}
    count = 0
    for uid in user_ids:
        count += 1
        ev = evidences.pop(uid)  # to free RAM
        #    logger.debug('training MEMM %d (user id: %s, dimensions: %d) ...', count, uid, ev[0])
        m = MEMM()
        try:
            m.fit(ev)
            memms[uid] = m
        except MemmException:
            logger.warn('evidences for user %s ignored due to insufficient data', uid)
        if count % 1000 == 0:
            logger.debug('[%s] %d memms trained', p_name, count)
            # logger.debug('\n{:25}{}\n'.format('piece of code', 'time (s)') +
            #              '{:25}{:.0f}\n'.format('all', times[0]) +
            #              '{:25}{:.0f}\n'.format('__decrease_dim', times[1]) +
            #              '{:25}{:.0f}\n'.format('> has_nonzero', times[6]) +
            #              '{:25}{:.0f}\n'.format('> orig_indexes', times[7]) +
            #              '{:25}{:.0f}\n'.format('> new_sequences', times[8]) +
            #              '{:25}{:.0f}\n'.format('all_obs_arr', times[2]) +
            #              '{:25}{:.0f}\n'.format('__get_related_pairs', times[3]) +
            #              '{:25}{:.0f}\n'.format('__create_matrices', times[13]) +
            #              '{:25}{:.0f}\n'.format('__calc_features', times[4]) +
            #              '{:25}{:.0f}\n'.format('iteration', times[5]) +
            #              '{:25}{:.0f}\n'.format('> __build_tpm', times[9]) +
            #              '{:25}{:.0f}\n'.format('> __build_expectation', times[10]) +
            #              '{:25}{:.0f}\n'.format('> __build_next_lambda', times[11]) +
            #              '{:25}{:.0f}\n'.format('> count_nonzero', times[12]))

    logger.debugv('[%s] training memms finished', p_name)
    return memms


def test_memms(children, parents_dic, observations, active_ids, memms, threshold):
    p_name = multiprocessing.current_process().name
    logger.debug('[%s] testing memms started', p_name)

    active_children = []

    j = 0
    for child_id in children:
        if child_id not in parents_dic:
            continue
        parents = parents_dic[child_id]
        # rel = mongodb.relations.find_one({'user_id': child_id}, {'_id': 0, 'parents': 1})
        # if rel is None:
        #     continue
        # parents = rel['parents']

        obs = observations[child_id]
        # logger.debug('child_id not in active_ids: %s, child_id in self.__memms: %s',
        #             child_id not in active_ids, child_id in self.__memms)

        # logger.debugv('child_id not in active_ids -> %d', child_id not in active_ids)
        # logger.debugv('str(child_id) in memms: -> %d', str(child_id) in memms)
        if child_id not in active_ids and child_id in memms:
            memm = memms[child_id]
            # logger.debug('predicting cascade ...')
            new_state, prob = memm.predict(obs, len(parents), threshold)
            logger.debugv('[%s] user id {%s} : probability {%f}', p_name, child_id, prob)

            if new_state == 1:
                active_children.append(child_id)
                active_ids.append(child_id)
                # logger.debug('\ta reshare predicted')

        j += 1
        if j % 100 == 0:
            logger.debugv('[%s] %d / %d of children iterated', p_name, j, len(children))

    del memms, children, parents_dic, observations, active_ids

    logger.debug('[%s] testing memms finished', p_name)
    return active_children


class MEMMModel():
    def __init__(self, project):
        self.project = project
        self.__memms = {}

    def __save_evidences(self, sequences):
        logger.info('saving MEMM evidences ...')
        self.project.save_param(sequences, MEMM_EVID_FILE_NAME, ParamTypes.JSON)
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
                    parents_count = len(rel['parents'])
                    logger.debug('extracting children ...')
                    # children = rel_dic[uid]['children']
                    children = rel['children']

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
                    rel = mongodb.relations.find_one({'user_id': uid}, {'_id': 0, 'parents': 1})
                    dim = len(rel['parents'])  # TODO: Collect counts at the beginning
                    new_evidences.setdefault(uid, [dim, []])
                    new_evidences[uid].sequences.append(cascade_seqs[uid])
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
        return big_userids, notbig_userids

    def __fit_multiproc(self, evidences):
        """
        Side effect: Clears the evidences dictionary.
        """
        user_ids = list(evidences.keys())
        process_count = multiprocessing.cpu_count()
        # process_count = 8
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

    def fit(self, train_set):
        """
        Train MEMM's for each user in training set.
        :param train_set:   cascade id's in training set
        :return:            self
        """
        t0 = time.time()

        # Divide evidences into some parts. Each time load a part from evidences and train the
        # corresponding MEMMS to avoid high RAM consumption.
        evidences = self.__prepare_evidences(train_set)

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

        logger.info("====== MEMM model training time: %.2f m", (time.time() - t0) / 60.0)

        return self

    def __predict_multiproc(self, children, parent_node, parents_dic, observations, active_ids, threshold, next_step):
        # process_count = multiprocessing.cpu_count()
        process_count = 2
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

    def predict(self, initial_tree, threshold=None, max_step=None):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :return:         Predicted tree
        """
        # ptimes = [0] * 4
        if not isinstance(initial_tree, CascadeTree):
            raise ValueError('tree must be CascadeTree')
        tree = initial_tree.copy()

        t0 = time.time()

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
                    # t = time.time()
                    relations = mongodb.relations.find({'user_id': {'$in': children}},
                                                       {'_id': 0, 'user_id': 1, 'parents': 1})
                    parents_dic = {rel['user_id']: rel['parents'] for rel in relations}
                    # ptimes[0] += time.time() - t

                    for child_id in children:
                        obs = observations.setdefault(child_id, 0)
                        if child_id not in parents_dic:
                            continue
                        parents = parents_dic[child_id]
                        index = parents.index(node_id)
                        obs |= 1 << (len(parents) - index - 1)
                        observations[child_id] = obs

                    if 1000 < len(children) < 200000:
                        self.__predict_multiproc(children, node, parents_dic, observations, active_ids, threshold,
                                                 next_step)
                    else:
                        memms_i = {uid: self.__memms[uid] for uid in children if uid in self.__memms}
                        act_children = test_memms(children, parents_dic, observations, active_ids, memms_i, threshold)
                        for child_id in act_children:
                            child = CascadeNode(child_id)
                            node.children.append(child)
                            next_step.append(child)
                            active_ids.append(child_id)

                i += 1
                logger.debug('%d / %d nodes of current step done', i, len(cur_step))
                # logger.debug('times: %s', [int(t) for t in ptimes[:4]])

            cur_step = next_step
            step_num += 1

        logger.debug('time1 = %.2f' % (time.time() - t0))

        return tree

    def __save_memms(self):
        uids = list(self.__memms.keys())
        save_dir = os.path.join(BASEPATH, 'data', self.project.project_name, 'memm')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, 'trained')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        lambda_all = {uid: self.__memms[uid].Lambda for uid in uids}
        np.savez(os.path.join(save_dir, 'labmda.npz'), **lambda_all)
        tpm_all = {uid: self.__memms[uid].TPM for uid in uids}
        np.savez(os.path.join(save_dir, 'tpm.npz'), **tpm_all)
        obs_arr_all = {uid: self.__memms[uid].all_obs_arr for uid in uids}
        np.savez(os.path.join(save_dir, 'all_obs_arr.npz'), **obs_arr_all)
        map_obs_index_all = {uid: self.__memms[uid].map_obs_index for uid in uids}
        with open(os.path.join(save_dir, 'map_obs_index.json'), 'w') as f:
            json.dump(map_obs_index_all, f)
        orig_indexes_all = {uid: self.__memms[uid].orig_indexes for uid in uids}
        with open(os.path.join(save_dir, 'orig_indexes.json'), 'w') as f:
            json.dump(orig_indexes_all, f)

    def __load_memms(self):
        self.__memms = {}
        load_dir = os.path.join(BASEPATH, 'data', self.project.project_name, 'memm', 'trained')
        lambda_all = np.load(os.path.join(load_dir, 'labmda.npz'))
        for uid in lambda_all:
            memm = MEMM()
            memm.Lambda = lambda_all[uid]
            self.__memms[uid] = memm
        tpm_all = np.load(os.path.join(load_dir, 'tpm.npz'))
        for uid in tpm_all:
            self.__memms[uid].TPM = tpm_all[uid]
        obs_arr_all = np.load(os.path.join(load_dir, 'all_obs_arr.npz'))
        for uid in obs_arr_all:
            self.__memms[uid].all_obs_arr = obs_arr_all[uid]
        with open(os.path.join(load_dir, 'map_obs_index.json')) as f:
            map_obs_index_all = json.load(f)
        for uid in map_obs_index_all:
            self.__memms[uid].map_obs_index = map_obs_index_all[uid]
        with open(os.path.join(load_dir, 'orig_indexes.json')) as f:
            orig_indexes_all = json.load(f)
        for uid in orig_indexes_all:
            self.__memms[uid].orig_indexes = orig_indexes_all[uid]
