import math
import random
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
from memm.asyncronizables import train_memms, extract_evidences
from db.exceptions import DataDoesNotExist
from db.managers import MEMMManager, DBManager, EvidenceManager
from db.reconnection import rerun_auto_reconnect, reconnect
from memm.memm import MEMM
from settings import logger
from utils.time_utils import Timer, TimeUnit, time_measure


class MEMMModel:
    def __init__(self, project):
        self.project = project
        self.__memms = {}

    @time_measure(level='debug')
    def __prepare_evidences(self, train_set):
        """
        Prepare the sequence of observations and states to train the MEMM models.
        :param train_set: list of training cascade id's
        :return: a dictionary of user id's to instances of MemmEvidence
        """
        evid_manager = EvidenceManager(self.project)

        try:
            logger.info('loading MEMM evidences ...')
            evidences = evid_manager.get_many()

        except DataDoesNotExist:
            logger.info('no evidences found! extraction started')
            evidences = {}  # dictionary of user id's to list of the sequences of ObsPair instances.
            act_seqs = self.project.load_or_extract_act_seq()

            logger.info('extracting sequences from %d cascades ...', len(train_set))

            process_count = min(settings.PROCESS_COUNT, len(train_set))
            pool = Pool(processes=process_count)
            step = int(math.ceil(float(len(train_set)) / process_count))
            results = []
            for j in range(0, len(train_set), step):
                cascade_ids = train_set[j: j + step]
                res = pool.apply_async(extract_evidences, (cascade_ids, act_seqs))
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

            # Delete evidences of totally inactive users since they will never be activated.
            inactives = self.__get_inactives(evidences)
            for uid in inactives:
                evidences.pop(uid)
            logger.info('Evidences of %d totally inactive users deleted since they have no state 1 ...', len(inactives))

            evid_manager.insert(evidences)
            evid_manager.create_index()

        return evidences

    def __get_inactives(self, evidences):
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

    def __separate_big_ev(self, evidences):
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

    def __fit_multiproc(self, evidences):
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
            res = pool.apply_async(train_memms, (evidences_i,))
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

    def __fit_by_evidences(self, train_set, multi_processed=False):
        evidences = self.__prepare_evidences(train_set)

        if multi_processed:
            single_process_ev = {}  # Evidences to train sequentially in a single process.
            multi_processed_ev = {}  # Evidences to train simultaneously in multiple processes.

            # Divide evidences into some parts. Each time load a part from evidences and train the
            # corresponding MEMMS to avoid high RAM consumption.
            logger.info('separating large and small evidences ...')
            large_ev_user_ids, small_ev_user_ids = self.__separate_big_ev(evidences)
            logger.info('%d large and %d small evidences considered', len(large_ev_user_ids), len(small_ev_user_ids))
            for uid in large_ev_user_ids:
                single_process_ev[uid] = evidences.pop(uid)  # to free RAM
            for uid in small_ev_user_ids:
                multi_processed_ev[uid] = evidences.pop(uid)  # to free RAM
            del evidences

            # Train not-big evidences if in multi-processed is True.
            # parts_count = 5
            # part_size = int(math.ceil(len(small_ev_user_ids) / parts_count))

            # for j in range(parts_count):
            #     logger.info('loading evidences of part %d of users', j + 1)
            #     part_j = small_ev_user_ids[j * part_size: (j + 1) * part_size]
            #     evidences_j = {}
            #     for uid in part_j:
            #         evidences_j[uid] = multi_processed_ev.pop(uid)  # to free RAM
            #     logger.info("training %d MEMM's related to part %d of users simultaneously ...", len(part_j), j + 1)
            #     memms = self.__fit_multiproc(evidences_j)
            #
            #     logger.info('inserting MEMMs into db ...')
            #     MEMMManager(self.project).insert(memms)
            # del memms, evidences_j

            memms = self.__fit_multiproc(multi_processed_ev)
            logger.info('inserting MEMMs into db ...')
            MEMMManager(self.project).insert(memms)
            del memms, multi_processed_ev

        else:
            single_process_ev = evidences

        """
        Train big evidences sequentially in a single process if multi_processed is True and all
        evidences otherwise.
        """
        logger.info('training %d MEMMs sequentially', len(single_process_ev))
        train_memms(single_process_ev, save_in_db=True, project=self.project)
        del single_process_ev

        logger.info('training MEMMs finished')

    @time_measure(level='debug')
    def fit(self, train_set, multi_processed=False):
        """
        Load MEMM's from DB if exist, otherwise train MEMM's for each user in training set.
        :param train_set:   cascade id's in training set
        :return:            self
        """
        manager = MEMMManager(self.project)

        # Train MEMMs if they are not saved in DB.
        if not manager.db_exists():
            logger.info('MEMMs do not exist in db. They will be trained')
            self.__fit_by_evidences(train_set, multi_processed)

        logger.info('loading MEMMs from db ...')
        try:
            self.__memms = manager.fetch_all()

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
        db = DBManager().db
        timers = [Timer(f'predict part {i}', level='debug', unit=TimeUnit.SECONDS, silent=True) for i in range(10)]

        # Dictionary of predicted trees related to thresholds: trees = { threshold1: tree1, threshold2: tree2, ... }
        trees = {thr: initial_tree.copy() for thr in thresholds}

        # Initialize values.
        cur_step_nodes = sorted(initial_tree.get_leaves(), key=lambda n: n.datetime)  # Set tree nodes as initial step.
        max_thr = max(thresholds)
        cur_step = [(node.user_id, max_thr) for node in cur_step_nodes]
        active_ids = set(initial_tree.node_ids())
        step_num = 1

        """
            Create dictionary of current observations of the nodes for each threshold:
            observations = {
                            threshold1: {user_id1: obs1, user_id2: obs2, ...},
                            threshold2: {user_id1: obs1, user_id2: obs2, ...},
                            ...
                            }
        """
        observations = {thr: {} for thr in thresholds}

        # Predict the cascade tree.
        # At each iteration find newly activated nodes based on MEMM probabilities and add them to the tree.
        while cur_step and (max_step is None or step_num <= max_step):
            logger.debug('predicting step %d ...', step_num)

            next_step = []

            relations = db.relations.find({'user_id': {'$in': [item[0] for item in cur_step]}},
                                          {'_id': 0, 'user_id': 1, 'children': 1})
            children_dic = {rel['user_id']: rel['children'] for rel in relations}

            parents_loaded = False
            parents_dic = None
            logger.debug('fetching parents ...')
            try:
                parents_dic = self.__fetch_children_parents(children_dic, db)
                parents_loaded = True
                logger.debug('done')
            except MemoryError:
                logger.debug('Low memory to fetch the parents of all children! trying to fetch them one by one.')

            i = 0
            for node_id, max_predicted_thr in cur_step:
                children = children_dic.pop(node_id, [])  # to get and free RAM

                if children:
                    logger.debug('user %s has %d children:', node_id, len(children))
                    # from utils.text_utils import columnize
                    # logger.debugv('\n' + columnize([str(child_id) for child_id in children], 4))

                    if not parents_loaded:
                        with timers[0]:
                            parents_dic = self.__fetch_parents_dic(children, db)

                    j = 0
                    for child_id in children:

                        if child_id not in active_ids:
                            memm = self.get_memm(child_id)

                            if memm is not None:

                                with timers[1]:
                                    # Update the observation of this child.
                                    self.update_observation(child_id, node_id, observations, memm, parents_dic,
                                                            thresholds, max_predicted_thr)

                                child_max_pred_thr = None
                                last_prob = None
                                last_obs = None

                                for thr in thresholds:
                                    if thr <= max_predicted_thr:
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

                                        if prob >= thr and trees[thr].get_node(node_id):
                                            trees[thr].add_child(node_id, child_id)
                                            child_max_pred_thr = thr
                                            logger.debugv('a reshare predicted %f >= %f', prob, thr)
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

            cur_step = next_step
            step_num += 1

        for timer in timers:
            if timer.sum != 0:
                timer.report_sum()

        return trees

    def update_observation(self, child_id: ObjectId, parent_id: ObjectId, observations: dict, child_memm: MEMM,
                           parents_dic: dict, thresholds: list, max_predicted_thr: float):
        """
        Update the observations the child node for all thresholds in such a way that reflects the activation of the
        parent given.
        :param child_id: child id
        :param parent_id: parent id
        :param observations: dictionary of the thresholds to observation dicts. Each observation dict maps child ids to their observation.
        :param child_memm: MEMM of the child node
        :param parents_dic: dictionary of node ids to their parent ids list.
        :param thresholds: list of thresholds
        :param max_predicted_thr: the maximum threshold in which the parent is predicted as activated. So the child observation is updated up to this threshold.
        """
        parents = parents_dic[child_id]
        index = parents.index(parent_id)
        try:
            decreased_ind = child_memm.orig_indexes.index(index)
        except ValueError:
            decreased_ind = None
        for thr in thresholds:
            if thr <= max_predicted_thr:
                obs_thr = observations[thr]
                obs = obs_thr.get(child_id, np.zeros(len(child_memm.orig_indexes), dtype=bool))
                if decreased_ind is not None:  # If the index found in the original indexes
                    obs[decreased_ind] = 1
                obs_thr[child_id] = obs
            else:
                break

    def __fetch_children_parents(self, children_dic, db):
        children = list(reduce(lambda x, y: x | y, (set(child_list) for child_list in children_dic.values())))
        relations = db.relations.find({'user_id': {'$in': children}}, {'_id': 0, 'user_id': 1, 'parents': 1})
        parents_dic = {rel['user_id']: rel['parents'] for rel in relations}
        return parents_dic

    def get_memm(self, user_id):
        if self.__memms:
            return self.__memms.get(user_id, None)
        else:
            return MEMMManager(self.project).fetch_one(user_id)

    @rerun_auto_reconnect
    def __fetch_parents_dic(self, children, db):
        relations = db.relations.find({'user_id': {'$in': children}},
                                      {'_id': 0, 'user_id': 1, 'parents': 1})
        parents_dic = {rel['user_id']: rel['parents'] for rel in relations}
        return parents_dic
