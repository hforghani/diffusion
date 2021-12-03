import math
import multiprocessing
import random
from multiprocessing.pool import Pool

from pympler.asizeof import asizeof

import settings
from cascade.models import CascadeNode, CascadeTree
from memm.asyncronizables import train_memms, test_memms, test_memms_eco, extract_evidences
from db.exceptions import DataDoesNotExist
from db.managers import MEMMManager, DBManager, EvidenceManager
from db.decorators import graceful_auto_reconnect
from settings import logger
from utils.time_utils import Timer


class MEMMModel:
    def __init__(self, project):
        self.project = project
        self.__memms = {}

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

            pool = Pool(processes=settings.PROCESS_COUNT)
            step = int(math.ceil(float(len(train_set)) / settings.PROCESS_COUNT))
            results = []
            for j in range(0, len(train_set), step):
                meme_ids = train_set[j: j + step]
                res = pool.apply_async(extract_evidences, (meme_ids, act_seqs))
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

            evid_manager.insert(evidences)
            evid_manager.create_index()

        return evidences

    def __separate_big_ev(self, evidences):
        big_userids = []
        notbig_userids = []
        big_threshold = 60000
        for uid in evidences:
            if asizeof(evidences[uid]['sequences']) > big_threshold:
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
        logger.debug('starting %d processes to train MEMMs', settings.PROCESS_COUNT)
        pool = Pool(processes=settings.PROCESS_COUNT)
        step = int(math.ceil(len(evidences) / settings.PROCESS_COUNT))
        results = []

        for i in range(settings.PROCESS_COUNT):
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
        logger.debug('assembling multiprocessed results of MEMM training ...')
        for res in results:
            memms_i = res.get()
            user_ids_i = list(memms_i.keys())
            for uid in user_ids_i:
                memms[uid] = memms_i.pop(uid)  # to free RAM
        logger.debug('assembling done')

        return memms

    def __fit_by_evidences(self, train_set):
        evidences = self.__prepare_evidences(train_set)

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
        del evidences

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
            memms = self.__fit_multiproc(evidences_j)

            logger.info('inserting MEMMs into db ...')
            MEMMManager(self.project).insert(memms)
        del memms, evidences_j

        # Train big evidences sequentially.
        logger.info('training %d big MEMMs sequentially', len(big_ev))
        train_memms(big_ev, save_in_db=True, project=self.project)
        del big_ev

        logger.info('training MEMMs finished')

    def fit(self, train_set):
        """
        Load MEMM's from DB if exist, otherwise train MEMM's for each user in training set.
        :param train_set:   cascade id's in training set
        :return:            self
        """
        logger.info('loading MEMMs from db ...')
        self.__memms = MEMMManager(self.project).fetch_all()

        # Train MEMMs if they are not saved in DB.
        if not self.__memms:
            logger.info('MEMMs do not exist in db. They will be trained')
            self.__fit_by_evidences(train_set)

            # logger.info('inserting MEMMs into db ...')
            self.__memms = MEMMManager(self.project).fetch_all()

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
                                   (children_i, parents_dic_i, observations_i, self.project, active_ids, threshold))
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

    def predict(self, initial_tree, threshold=None, max_step=None, multiprocessed=True):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :return:         Predicted tree
        """
        db = DBManager().db
        ch_timer = Timer('get children', level='debug')
        p_timer = Timer('get parents', level='debug')
        obs_timer = Timer('observations', level='debug')
        m_timer = Timer('MEMM predict', level='debug')

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
                relations = db.relations.find({'user_id': {'$in': [n.user_id for n in cur_step]}},
                                              {'_id': 0, 'user_id': 1, 'children': 1})
                children_dic = {rel['user_id']: rel['children'] for rel in relations}

            i = 0
            for node in cur_step:
                node_id = node.user_id
                children = children_dic.pop(node_id, [])  # to get and free RAM
                # rel = db.relations.find_one({'user_id': node_id}, {'_id': 0, 'children': 1})
                # children = rel['children'] if rel is not None else []

                if children:
                    logger.debug('user %s has %d children:', node_id, len(children))
                    # logger.debugv('\n' + columnize([str(child_id) for child_id in children], 4))

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
                            parents_dic = self.__get_parents_dic(children, db)

                        with obs_timer:
                            for child_id in children:
                                obs = observations.setdefault(child_id, 0)
                                parents = parents_dic[child_id]
                                index = parents.index(node_id)
                                obs |= 1 << (len(parents) - index - 1)
                                observations[child_id] = obs

                        with m_timer:
                            if len(children) > 150000 and multiprocessed:
                                self.__predict_multiproc_eco(children, node, parents_dic, observations, active_ids,
                                                             threshold, next_step)
                            elif len(children) > 1000 and multiprocessed:
                                self.__predict_multiproc(children, node, parents_dic, observations, active_ids,
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

    @graceful_auto_reconnect
    def __get_parents_dic(self, children, db):
        relations = db.relations.find({'user_id': {'$in': children}},
                                      {'_id': 0, 'user_id': 1, 'parents': 1})
        parents_dic = {rel['user_id']: rel['parents'] for rel in relations}
        return parents_dic
