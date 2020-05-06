import time
from cascade.models import CascadeNode, CascadeTree, ParamTypes
from memm.memm import MEMM, MemmException, times
from neo4j.models import Neo4jGraph
from settings import logger


#MEMM_EVID_FILE_NAME = 'memm/evidence'
MEMM_EVID_FILE_NAME = 'memm/evidence-5d88f41e86887707d4526076'


class MEMMModel():
    def __init__(self, project):
        self.project = project
        self.__memms = {}
        self.__graph = Neo4jGraph('User')

    def __save_evidences(self, sequences, log=0):
        if log > 0:
            logger.info('saving MEMM evidences ...')
        self.project.save_param(sequences, MEMM_EVID_FILE_NAME, ParamTypes.JSON)
        if log > 0:
            logger.info('done')

    def __load_evidences(self, log=0):
        if log > 0:
            logger.info('loading MEMM evidences ...')
        tsets = self.project.load_param(MEMM_EVID_FILE_NAME, ParamTypes.JSON)
        return tsets

    def __add_and_save_evidences(self, new_evidences, log):
        try:
            evidences = self.__load_evidences(log)
        except FileNotFoundError:
            evidences = {}

        if new_evidences:
            if log > 0:
                logger.info('integrating MEMM evidences ...')
            for uid in new_evidences:
                if uid not in evidences:
                    evidences[uid] = new_evidences[uid]
                else:
                    evidences[uid][1].extend(new_evidences[uid][1])
            self.__save_evidences(evidences, log)

        return evidences

    def __prepare_evidences(self, train_set, log=0):
        """
        Prepare the sequence of observations and states to train the MEMM models.
        :param train_set: list of training cascade id's
        :param log: log level
        :return: a dictionary of user id's to instances of MemmEvidence
        """
        act_seqs = self.project.load_or_extract_act_seq()

        try:
            evidences = self.__load_evidences(log)

        except FileNotFoundError:
            logger.info('no evidences found! extraction started')
            count = 0
            new_evidences = {}  # dictionary of user id's to list of the sequences of ObsPair instances which are not saved yet.
            cascade_seqs = {}   # dictionary of user id's to the sequences of ObsPair instances for this current cascade

            if log > 0:
                logger.info('extracting sequences from %d cascades ...', len(train_set))

            # Iterate each activation sequence and extract sequences of (observation, state) for each user
            for cascade_id in train_set:
                act_seq = act_seqs[cascade_id]
                observations = {}   # current observation of each user
                activated = set()   # set of current activated users
                i = 0
                if log > 1:
                    logger.info('cascade %d with %d users ...', count + 1, len(act_seq.users))

                for uid in act_seq.users:  # Notice users are sorted by activation time.
                    activated.add(uid)
                    parents_count = self.__graph.parents_count(uid)
                    if log > 1:
                        logger.info('extracting children ...')
                    children = self.__graph.children(uid)

                    # Put the last observation with state 1 in the sequence of (observation, state).
                    if parents_count:
                        observations.setdefault(uid, 0)  # initial observation: 0000000
                        cascade_seqs.setdefault(uid, [])
                        if cascade_seqs[uid]:
                            obs = cascade_seqs[uid][-1][0]
                            del cascade_seqs[uid][-1]
                            cascade_seqs[uid].append((obs, 1))

                    if log > 1 and children:
                        logger.info('iterating on %d children ...', len(children))

                    # Set the related coefficient of the children observations equal to 1.
                    j = 0
                    for child in children:
                        child_parents = self.__graph.get_or_fetch_parents(child)
                        obs = observations.setdefault(child, 0)
                        index = child_parents.index(uid)
                        obs |= 1 << (len(child_parents) - index - 1)
                        observations[child] = obs
                        if child not in activated:
                            cascade_seqs.setdefault(child, [(0, 0)])
                            cascade_seqs[child].append((obs, 0))
                        j += 1
                        if log > 1 and j % 1000 == 0:
                            logger.info('%d children done', j)

                    i += 1
                    if log > 1:
                        logger.info('%d users done', i)

                count += 1
                if log > 0 and count % 1000 == 0:
                    logger.info('%d cascades done', count)

                # Add current sequence of pairs (observation, state) to the MEMM evidences.
                logger.info('adding sequences of current cascade ...')
                for uid in cascade_seqs:
                    dim = self.__graph.parents_count(uid)
                    new_evidences.setdefault(uid, [dim, []])
                    new_evidences[uid].sequences.append(cascade_seqs[uid])
                cascade_seqs = {}

                if len(act_seq.users) > 1000:
                    self.__add_and_save_evidences(new_evidences, log)
                    new_evidences = {}

            evidences = self.__add_and_save_evidences(new_evidences, log)

        self.__graph = Neo4jGraph('User')  # To clear the cache.
        return evidences

    def fit(self, train_set, log=0):
        """
        Train MEMM's for each user in training set.
        :param train_set:   cascade id's in training set
        :return:            self
        """
        t0 = time.time()
        evidences = self.__prepare_evidences(train_set, log)
        user_ids = list(evidences.keys())

        # Train a MEMM for each user.
        if log > 0:
            logger.info("training %d MEMM's ...", len(evidences))
        count = 0
        for uid in user_ids:
            count += 1
            ev = evidences[uid]
            #if log > 0:
            #    logger.info('training MEMM %d (user id: %s, dimensions: %d) ...', count, uid, ev[0])
            m = MEMM()
            try:
                t1 = time.time()
                m.fit(ev, log)
                times[0] += time.time() - t1
                self.__memms[uid] = m
                del evidences[uid]  # to free RAM
            except MemmException:
                logger.warn('evidences for user %s ignored due to insufficient data', uid)
            if count % 1000 == 0:
                logger.debug('%d MEMM models trained', count)
                logger.debug('times : %s', [int(t) for t in times])

        if log > 0:
            logger.info("====== MEMM model training time: %.2f m", (time.time() - t0) / 60.0)
        return self

    def predict(self, initial_tree, threshold=None, max_step=None):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :param log:      Log in console if True else does not log.
        :return:         Predicted tree
        """
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
            logger.debug('\t predicting step %d ...', step_num)

            next_step = []

            for node in cur_step:
                uid = node.user_id
                children = self.__graph.children(uid)

                for child_id in children:
                    parents = self.__graph.get_or_fetch_parents(child_id)
                    obs = observations.setdefault(child_id, 0)
                    index = parents.index(uid)
                    obs |= 1 << (len(parents) - index - 1)
                    observations[child_id] = obs

                    if child_id not in active_ids and child_id in self.__memms:
                        memm = self.__memms[child_id]
                        logger.debug('predicting cascade ...')
                        new_state = memm.predict(obs, len(parents), threshold)

                        if new_state == 1:
                            child = CascadeNode(child_id)
                            node.children.append(child)
                            next_step.append(child)
                            active_ids.append(child_id)
                            logger.debug('\ta reshare predicted')
            cur_step = next_step
            step_num += 1

        logger.debug('time1 = %.2f' % (time.time() - t0))

        return tree
