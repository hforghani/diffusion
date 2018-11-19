import logging
import time
from cascade.models import CascadeNode, CascadeTree, ParamTypes
from memm.memm import MEMM

logger = logging.getLogger('memm.models')


class MEMMModel():
    def __init__(self, project):
        self.project = project
        self.__memms = {}
        self.__parents = {}
        self.__children = {}

    def fit(self, train_set, log=False):
        """
        Train MEMM's for each user in training set.
        :param train_set:   meme id's in training set
        :return:            self
        """
        t0 = time.time()
        graph, act_seqs = self.project.load_or_extract_graph_seq()

        # Extract all user id's in training set.
        user_ids = set()
        for meme_id in train_set:
            user_ids.update(act_seqs[meme_id].users)

        # Create dictionary of parents and children of each node.
        if log:
            logger.info('collecting parents and children data ...')
        graph_nodes = set(graph.nodes())
        self.__parents = {uid: list(graph.predecessors(uid)) if uid in graph_nodes else []
                          for uid in user_ids}
        self.__children = {uid: list(graph.successors(uid)) if uid in graph_nodes else []
                           for uid in user_ids}

        try:
            sequences = self.project.load_param('seq-obs-state', ParamTypes.JSON)

        except:
            sequences = {}  # sequences of (observation, state) for each user
            count = 0

            # Iterate each activation sequence and extract sequences of (observation, state) for each user
            if log:
                logger.info('extracting sequences from memes ...')
            for meme_id in train_set:
                act_seq = act_seqs[meme_id]
                observations = {}
                activated = set()
                meme_seqs = {}

                for uid in act_seq.users:   # Notice users are sorted by activation time.
                    activated.add(uid)
                    if self.__parents[uid]:
                        u_obs = observations.setdefault(uid, [0] * len(self.__parents[uid]))
                        meme_seqs.setdefault(uid, [])
                        meme_seqs[uid].append((''.join([str(o) for o in u_obs]), 1))
                    for child in self.__children[uid]:
                        obs = observations.setdefault(child, [0] * len(self.__parents[child]))
                        if child not in activated:
                            meme_seqs.setdefault(child, [])
                            meme_seqs[child].append((''.join([str(o) for o in obs]), 0))
                        index = self.__parents[child].index(uid)
                        obs[index] = 1
                for uid in meme_seqs:
                    if len(meme_seqs[uid]) > 1:
                        sequences.setdefault(uid, [])
                        sequences[uid].append(meme_seqs[uid])
                count += 1
                if count % 1000 == 0:
                    if log:
                        logger.info('%d memes done', count)

            self.project.save_param(sequences, 'seq-obs-state', ParamTypes.JSON)

        # Train a MEMM for each user.
        if log:
            logger.info("training %d MEMM's ...", len(sequences))
        count = 0
        for uid, seq in sequences.items():
            count += 1
            uid = int(uid)
            obs_dim = len(self.__parents[uid])
            if log:
                logger.info('training MEMM %d (user id: %d, dimensions: %d) ...', count, uid, obs_dim)
            m = MEMM().fit(seq, obs_dim)
            self.__memms[uid] = m

        if log:
            logger.info("====== MEMM model training time: %.2f m", (time.time() - t0) / 60.0)
        return self

    def predict(self, initial_tree, threshold=None, log=False):
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
        cur_step = sorted(tree.nodes(), key=lambda n: n.datetime)  # Set tree nodes as initial step.
        active_ids = initial_tree.node_ids()

        # Create dictionary of current observations of the nodes.
        observations = {uid: [0] * len(self.__parents.get(uid, [])) for uid in active_ids}

        # Predict the cascade tree.
        # At each iteration find newly activated nodes based on MEMM probabilities and add them to the tree.
        while cur_step:
            next_step = []
            for node in cur_step:
                uid = node.user_id
                for child_id in self.__children.get(uid, []):
                    obs = observations.setdefault(child_id, [0] * len(self.__parents[child_id]))
                    index = self.__parents[child_id].index(uid)
                    obs[index] = 1
                    if child_id not in active_ids and child_id in self.__memms:
                        memm = self.__memms[child_id]
                        obs_str = ''.join([str(o) for o in obs])
                        new_state = memm.predict(obs_str, threshold)
                        if new_state == 1:
                            child = CascadeNode(child_id)
                            node.children.append(child)
                            next_step.append(child)
                            active_ids.append(child_id)
                            if log:
                                logger.info('\ta reshare predicted')
            cur_step = next_step

        if log:
            logger.info('time1 = %.2f' % (time.time() - t0))

        return tree
