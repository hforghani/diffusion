import json
import os
import time
from cascade.models import CascadeNode, CascadeTree, ParamTypes
from memm.memm import MEMM, MemmException, times
from neo4j.models import Neo4jGraph
import numpy as np
from settings import logger, BASEPATH, mongodb


MEMM_EVID_FILE_NAME = 'memm/evidence'
# MEMM_EVID_FILE_NAME = 'memm/evidence-5d88f41e86887707d4526076'


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
        tsets = self.project.load_param(MEMM_EVID_FILE_NAME, ParamTypes.JSON)
        return tsets

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
            cascade_seqs = {}   # dictionary of user id's to the sequences of ObsPair instances for this current cascade

            logger.info('extracting sequences from %d cascades ...', len(train_set))

            # Iterate each activation sequence and extract sequences of (observation, state) for each user
            for cascade_id in train_set:
                act_seq = act_seqs[cascade_id]
                observations = {}   # current observation of each user
                activated = set()   # set of current activated users
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
                    dim = len(rel['parents']) #TODO: Collect counts at the beginning
                    new_evidences.setdefault(uid, [dim, []])
                    new_evidences[uid].sequences.append(cascade_seqs[uid])
                cascade_seqs = {}

                if len(act_seq.users) > 1000:
                    self.__add_and_save_evidences(new_evidences)
                    new_evidences = {}

            evidences = self.__add_and_save_evidences(new_evidences)

        return evidences

    def fit(self, train_set):
        """
        Train MEMM's for each user in training set.
        :param train_set:   cascade id's in training set
        :return:            self
        """
        #try:
        #    logger.info('loading trained MEMMs ...')
        #    self.__load_memms()
        #
        #except FileNotFoundError:
        logger.info('MEMMs not found. training ...')
        t0 = time.time()

        evidences = self.__prepare_evidences(train_set)
        user_ids = list(evidences.keys())

        # Train a MEMM for each user.
        logger.info("training %d MEMM's ...", len(evidences))
        count = 0
        for uid in user_ids:
            count += 1
            ev = evidences[uid]
            #    logger.debug('training MEMM %d (user id: %s, dimensions: %d) ...', count, uid, ev[0])
            m = MEMM()
            try:
                m.fit(ev)
                self.__memms[uid] = m
                del evidences[uid]  # to free RAM
            except MemmException:
                logger.warn('evidences for user %s ignored due to insufficient data', uid)
            if count % 10000 == 0:
                logger.debug('%d MEMM models trained', count)
                logger.debug('\n{:25}{}\n'.format('piece of code', 'time (s)') +
                             '{:25}{:.0f}\n'.format('all', times[0]) +
                             '{:25}{:.0f}\n'.format('__decrease_dim', times[1]) +
                             '{:25}{:.0f}\n'.format('> has_nonzero', times[6]) +
                             '{:25}{:.0f}\n'.format('> orig_indexes', times[7]) +
                             '{:25}{:.0f}\n'.format('> new_sequences', times[8]) +
                             '{:25}{:.0f}\n'.format('all_obs_arr', times[2]) +
                             '{:25}{:.0f}\n'.format('__get_related_pairs', times[3]) +
                             '{:25}{:.0f}\n'.format('__create_matrices', times[13]) +
                             '{:25}{:.0f}\n'.format('__calc_features', times[4]) +
                             '{:25}{:.0f}\n'.format('iteration', times[5]) +
                             '{:25}{:.0f}\n'.format('> __build_tpm', times[9]) +
                             '{:25}{:.0f}\n'.format('> __build_expectation', times[10]) +
                             '{:25}{:.0f}\n'.format('> __build_next_lambda', times[11]) +
                             '{:25}{:.0f}\n'.format('> count_nonzero', times[12]))

        # logger.info('saving trained MEMMs ...')
        #self.__save_memms()
        logger.info("====== MEMM model training time: %.2f m", (time.time() - t0) / 60.0)

        return self

    def predict(self, initial_tree, threshold=None, max_step=None):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :return:         Predicted tree
        """
        ptimes = [0] * 4
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

            # relations = mongodb.relations.find({'user_id': {'$in': [n.user_id for n in cur_step]}}, {'_id':0, 'user_id':1, 'children':1})
            # children_dic = {rel['user_id']: rel['children'] for rel in relations}

            i = 0
            for node in cur_step:
                uid = node.user_id
                # children = children_dic.get(uid, [])
                rel = mongodb.relations.find_one({'user_id': uid}, {'_id': 0, 'children': 1})
                children = rel['children'] if rel is not None else []

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

                elif threshold != 1:
                    # relations = mongodb.relations.find({'user_id': {'$in': children}}, {'_id':0, 'user_id':1, 'parents':1})
                    # parents_dic = {rel['user_id']: rel['parents'] for rel in relations}

                    j = 0
                    for child_id in children:
                        t = time.time()
                        # if child_id not in parents_dic:
                        #     continue
                        # parents = parents_dic[child_id]
                        rel = mongodb.relations.find_one({'user_id': child_id}, {'_id': 0, 'parents': 1})
                        if rel is None:
                            continue
                        parents = rel['parents']

                        ptimes[0] += time.time() - t
                        t = time.time()
                        obs = observations.setdefault(child_id, 0)
                        index = parents.index(uid)
                        obs |= 1 << (len(parents) - index - 1)
                        observations[child_id] = obs
                        ptimes[1] += time.time() - t
                        #logger.debug('child_id not in active_ids: %s, child_id in self.__memms: %s',
                        #             child_id not in active_ids, child_id in self.__memms)

                        if child_id not in active_ids and str(child_id) in self.__memms:
                            t = time.time()
                            memm = self.__memms[str(child_id)]
                            #logger.debug('predicting cascade ...')
                            new_state = memm.predict(obs, len(parents), threshold)

                            if new_state == 1:
                                child = CascadeNode(child_id)
                                node.children.append(child)
                                next_step.append(child)
                                active_ids.append(child_id)
                                #logger.debug('\ta reshare predicted')

                            ptimes[2] += time.time() - t

                        j += 1
                        if j % 100 == 0:
                            logger.debugv('%d / %d of children iterated', j, len(children))

                i += 1
                logger.debug('%d / %d nodes of current step done', i, len(cur_step))
                logger.debug('times: %s', [int(t) for t in ptimes[:3]])

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
