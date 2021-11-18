# -*- coding: utf-8 -*-
from datetime import timedelta
import json
import os
import random
import time

from anytree import Node, RenderTree
from bson.objectid import ObjectId
from networkx import DiGraph, read_adjlist, relabel_nodes, write_adjlist
import numpy as np
from pymongo.errors import CursorNotFound, DocumentTooLarge

import settings
from db.managers import DBManager
from settings import logger
from utils.numpy_utils import load_sparse, save_sparse, save_sparse_list, load_sparse_list
from utils.os_utils import mkdir_rec
from utils.time_utils import str_to_datetime, DT_FORMAT, Timer, time_measure


class CascadeNode(object):
    def __init__(self, user_id=None, datetime=None, post_id=None, parent_id=None):
        self.user_id = user_id
        self.datetime = datetime
        self.post_id = post_id
        self.parent_id = parent_id
        self.children = []

    def get_dict(self):
        """
        Get dictionary of the object.
        """
        return {'user_id': str(self.user_id),
                'datetime': self.datetime,
                'post_id': str(self.post_id),
                'parent_id': str(self.parent_id) if self.parent_id is not None else None,
                'children': [node.get_dict() for node in self.children]}

    def from_dict(self, node_dict):
        """
        Set attributes from dictionary.
        node_dict: dictionary of node,
        parent_id: parent node id,
        users_map: dictionary of mapping from user id's to users.
        """
        self.user_id = ObjectId(node_dict['user_id'])
        self.datetime = node_dict['datetime']
        self.post_id = ObjectId(node_dict['post_id'])
        self.parent_id = ObjectId(node_dict['parent_id']) if node_dict['parent_id'] is not None else None
        self.children = [CascadeNode().from_dict(node) for node in node_dict['children']]
        return self

    def copy(self, max_depth=None):
        """
        Get an independent copy of the object.
        """
        node = CascadeNode(self.user_id, self.datetime, self.post_id, self.parent_id)
        if max_depth is None or max_depth > 0:
            max_depth = max_depth - 1 if max_depth is not None else None
            node.children = [child.copy(max_depth) for child in self.children]
        return node

    def depth(self):
        depth = 0
        for node in self.children:
            depth = max(depth, node.depth() + 1)
        return depth

    def __create_anytree_node(self, digest=False):
        if not digest:
            node = Node('{}({})'.format(self.user_id, self.post_id))
        else:
            node = Node(str(self.user_id))
        for child in self.children:
            child_node = child.__create_anytree_node(digest)
            child_node.parent = node
        return node

    def render(self, digest=False):
        node = self.__create_anytree_node(digest)
        lines = []
        for pre, fill, node in RenderTree(node):
            lines.append('%s%s' % (pre, node.name))
        return '\n'.join(lines)


class CascadeTree(object):
    roots = []
    depth = 0

    def __init__(self, roots=None):
        if roots is not None:
            if not isinstance(roots, list):
                raise ValueError('tree must be a list of root nodes')
            self.roots = roots
            self.depth = self.__calc_depth()

    @classmethod
    def extract_cascade(cls, meme_id):
        with Timer('TREE: fetching posts', 'debug'):
            # Fetch posts related to the meme and reshares.
            if isinstance(meme_id, str):
                meme_id = ObjectId(meme_id)
            db = DBManager().db
            post_ids = db.postmemes.distinct('post_id', {'meme_id': meme_id})
            posts = db.posts.find({'_id': {'$in': post_ids}}, {'url': 0}).sort('datetime')

            user_ids = list(set([p['author_id'] for p in posts]))
            reshares = db.reshares.find({'post_id': {'$in': post_ids}, 'reshared_post_id': {'$in': post_ids}}) \
                .sort('datetime')

        with Timer('TREE: creating nodes', 'debug'):
            # Create nodes for the users.
            nodes = {}
            visited = {uid: False for uid in user_ids}  # Set visited True if the node has been visited.
            for user_id in user_ids:
                nodes[user_id] = CascadeNode(user_id)

        with Timer('TREE: creating diffusion edges'):
            # Create diffusion edge if a user reshares to another for the first time. Note that reshares are sorted by time.
            logger.debug('TREE: reshares count = %d' % reshares.count())
            roots = []
            for reshare in reshares:
                child_id = reshare['user_id']
                parent_id = reshare['ref_user_id']
                if child_id == parent_id:
                    continue  # Continue if the reshare is between same users.
                parent = nodes[parent_id]

                if not visited[parent_id]:  # It is a root
                    parent.post_id = reshare['reshared_post_id']
                    parent.datetime = reshare['ref_datetime'].strftime(DT_FORMAT) if reshare['ref_datetime'] else None
                    visited[parent_id] = True
                    roots.append(parent)

                if not visited[child_id]:  # Any other node
                    child = nodes[child_id]
                    parent.children.append(child)
                    child.parent_id = parent_id
                    child.post_id = reshare['post_id']
                    child.datetime = reshare['datetime'].strftime(DT_FORMAT)
                    visited[child_id] = True

        with Timer('TREE: Adding single nodes'):
            # Add users with no diffusion edges as single nodes.
            t1 = time.time()
            first_posts = {}
            posts.rewind()

            for post in posts:
                if post['author_id'] not in first_posts:
                    first_posts[post['author_id']] = post
            for uid, node in nodes.items():
                if not visited[uid]:
                    post = first_posts[uid]
                    node.datetime = post['datetime'].strftime(DT_FORMAT) if post['datetime'] else None
                    node.post_id = post['_id']
                    roots.append(node)

        return cls(roots)

    def get_dict(self):
        return [node.get_dict() for node in self.roots]

    @classmethod
    def from_dict(cls, tree_dict):
        roots = []
        for node in tree_dict:
            roots.append(CascadeNode().from_dict(node))
        return cls(roots)

    def max_datetime(self, node=None):
        """
        Get maximum datetime of nodes.
        """
        if node is None:
            max_dt = None
            if self.roots and self.roots[0].datetime is not None:
                max_dt = str_to_datetime(self.roots[0].datetime)
            for root in self.roots:
                max_dt = max(max_dt, self.max_datetime(root))
        else:
            max_dt = str_to_datetime(node.datetime) if node.datetime is not None else None
            for child in node.children:
                max_dt = max(max_dt, self.max_datetime(child))
        return max_dt

    def get_leaves(self, node=None):
        leaves = []
        if node is None:
            for root in self.roots:
                leaves.extend(self.get_leaves(root))
        else:
            if not node.children:
                leaves.append(node)
            else:
                for child in node.children:
                    leaves.extend(self.get_leaves(child))
        return leaves

    def node_ids(self, max_depth=None):
        return [node.user_id for node in self.nodes(max_depth=max_depth)]

    def nodes(self, node=None, max_depth=None):
        nodes_list = []
        if node is None:
            for node in self.roots:
                nodes_list.extend(self.nodes(node, max_depth))
        else:
            nodes_list = [node]
            if max_depth is None or max_depth > 0:
                max_depth = max_depth - 1 if max_depth is not None else None
                for child in node.children:
                    nodes_list.extend(self.nodes(child, max_depth))
        return nodes_list

    def edges(self, node=None):
        edges_list = []
        if node is None:
            for root in self.roots:
                edges_list.extend(self.edges(root))
        else:
            parent_id = node.user_id
            edges_list = [(parent_id, child.user_id) for child in node.children]
            for child in node.children:
                edges_list.extend(self.edges(child))
        return edges_list

    def copy(self, max_depth=None):
        tree_copy = [root.copy(max_depth) for root in self.roots]
        return CascadeTree(tree_copy)

    def __calc_depth(self):
        depth = 0
        for node in self.roots:
            depth = max(depth, node.depth())
        return depth

    def render(self, digest=False):
        return '\n'.join([root.render(digest) for root in self.roots])


class ActSequence(object):
    """
    Activation Sequence: (u1, t1), (u2, t2), ..., (u_n, t_n)
    """

    def __init__(self, users=None, times=None, max_t=None):
        # Suppose times are sorted and users[i] corresponds to times[i]
        self.users = users if users is not None else []
        self.times = times if times is not None else []
        if users and times:
            self.user_times = {users[i]: times[i] for i in range(len(users))}
        self.max_t = max_t
        self.rond_set = None

    def users_before_time(self, datetime):
        f = 0
        l = len(self.times) - 1
        while f != l:
            m = (f + l) / 2
            if datetime < self.times[m]:
                l = m
            else:
                f = m
        return self.users[:f + 1]

    def users_before_user(self, user_id):
        index = self.users.index(user_id)
        return self.users[:index]

    def get_rond_set(self, graph):
        # Return set of non-active nodes with at least one active parent node for each.
        if self.rond_set is None:
            result = set()
            for uid in self.users:
                if uid in graph.nodes():
                    result.update(set(graph.successors(uid)))
            result = result - set(self.users)
            self.rond_set = result
        return self.rond_set

    def get_active_parents(self, uid, graph):
        rond_set = self.get_rond_set(graph)
        parents = set(graph.predecessors(uid)) if uid in graph.nodes() else set()
        if uid in rond_set:
            active_parents = parents & set(self.users)
        else:
            active_parents = parents & set(self.users_before_user(uid))
        return active_parents


class AsLT(object):
    def __init__(self, project):
        """
        w_param_name and r_param_name must be set in children.
        :param project:
        """
        self.project = project
        self.init_tree = None
        self.max_delay = 10000
        self.probabilities = {}  # dictionary of node id's to probabilities of activation
        self.user_ids = None
        self.users_map = None

        self.w = self.project.load_param(self.w_param_name, ParamTypes.SPARSE)
        # logger.info('sum of columns:')
        # for i in range(self.w.shape[1]):
        #     if self.w[:, i].nnz:
        #         logger.info('%d: %f', i, self.w[:, i].sum())
        self.w = self.w.tocsr()
        self.r = self.project.load_param(self.r_param_name, ParamTypes.ARRAY)

    def fit(self):
        pass

    # @profile
    def predict(self, initial_tree, threshold=None, max_step=None, user_ids=None, users_map=None):
        """
        Predict activation cascade in the future starting from initial nodes in self.tree.
        Set the final tree again in self.tree.
        :param initial_tree:    Initial tree of activated nodes
        :param threshold:       Threshold of activation probability. IF None, threshold is sampled randomly.
        :param user_ids:        List of possible users for activation. All of users if value is None.
        :return:                Returns self.tree
        """
        if not isinstance(initial_tree, CascadeTree):
            raise ValueError('tree must be a CascadeTree')
        if not hasattr(self, 'w') or not hasattr(self, 'r'):
            raise Exception('No w and r parameters found. Train the AsLT first.')
        tree = initial_tree.copy()

        # Initialize values.
        cur_step = sorted(tree.get_leaves(), key=lambda n: n.datetime)  # Set tree nodes as initial step.
        activated = tree.node_ids()
        self.probabilities = {}
        db = DBManager().db

        if user_ids is None or users_map is None:
            user_ids = [u['_id'] for u in db.users.find({}, ['_id']).sort('_id')]
            users_map = {user_ids[i]: i for i in range(len(user_ids))}
        self.user_ids = user_ids
        self.users_map = users_map

        thresholds = {}

        # Iterate on steps. For each step try to activate other nodes.
        step = 0
        while cur_step and (max_step is None or step < max_step):
            t0 = time.time()
            step += 1
            logger.debug('step {} on {} users ...'.format(step, len(cur_step)))

            next_step = []

            # Iterate on current step nodes to check if a child will be activated.
            for node in cur_step:
                u = node.user_id  # sender user id
                u_i = self.users_map[u]
                # w_u = np.squeeze(np.array(w[u_i, :].todense()))  # weights of the children of u
                w_u = self.w[u_i, :]
                if w_u.nnz:
                    logger.debugv('weights of user %d :\n' + '\n'.join(
                        ['{} : {}'.format(w_u.indices[i], w_u.data[i]) for i in range(w_u.nnz)]), u_i)

                # Iterate on children of u
                # for v_i in np.nonzero(w_u)[0]:
                for i in range(w_u.nnz):
                    v_i = w_u.indices[i]
                    v = user_ids[v_i]  # receiver (child) user id
                    if v in activated:
                        logger.debugv('user %d is already activated', v_i)
                        continue
                    if v not in self.probabilities:
                        self.probabilities[v] = 0
                        # self.probabilities[v] += w_u[v_i]
                    self.probabilities[v] += w_u.data[i]
                    logger.debugv('probability of user %d = %f', v_i, self.probabilities[v])

                    # Set the threshold or sample it randomly if None.
                    if threshold is None:
                        if v_i in thresholds:
                            thresh = thresholds[v_i]
                        else:
                            thresh = random.random()
                            thresholds[v_i] = thresh
                    else:
                        thresh = threshold

                    if self.probabilities[v] >= thresh:
                        # Get delay parameter.
                        delay_param = self.r[v_i]

                        # Sample delay from exponential distribution and calculate the receive time.
                        delay = 1 / delay_param if delay_param > 0 else self.max_delay  # in days
                        if delay > self.max_delay: delay = self.max_delay

                        # if delay_param > 0:
                        #     delay = np.random.exponential(delay_param)  # in days
                        # else:
                        #     logger.warn('delay param = {}'.format(delay_param))
                        #     delay = 1000    # a very large delay!

                        send_dt = str_to_datetime(node.datetime)
                        try:
                            receive_dt = send_dt + timedelta(days=delay)
                        except:
                            logger.debug('send_dt: {}, delay: {}'.format(send_dt, delay))
                            raise

                        # Add it to the tree.
                        child = CascadeNode(v, datetime=receive_dt.strftime(DT_FORMAT))
                        # child = CascadeNode(v)
                        node.children.append(child)
                        activated.append(v)
                        next_step.append(child)
                        logger.debugv('a reshare predicted: prob (%f) >= thresh (%f)', self.probabilities[v], thresh)
            cur_step = sorted(next_step, key=lambda n: n.datetime)
            logger.debug('step %d done, time = %.2f s' % (step, time.time() - t0))

        return tree


class IC(object):
    def __init__(self, project):
        self.project = project
        self.init_tree = None
        self.p_param_name = 'p'
        self.r_param_name = 'r'
        self.probabilities = {}  # dictionary of node id's to probabilities of activation
        self.user_map = None

    def fit(self):
        pass

    def predict(self, initial_tree, threshold=None, user_ids=None):
        if not isinstance(initial_tree, CascadeTree):
            raise ValueError('tree must be CascadeTree')
        tree = initial_tree.copy()
        now = tree.max_datetime()  # Find the datetime of now.

        # Initialize values.
        t0 = time.time()
        cur_step = sorted(tree.nodes(), key=lambda n: n.datetime)  # Set tree nodes as initial step.
        activated = tree.nodes()
        if self.user_map is None:
            if user_ids is None:
                db = DBManager().db
                user_ids = [u['_id'] for u in db.users.find({}, ['_id']).sort('_id')]
            self.user_map = {user_ids[i]: i for i in range(len(user_ids))}
        logger.debug('time1 = %.2f' % (time.time() - t0))

        # Get diffusion probabilities and delay vectors.
        t0 = time.time()
        p = self.project.load_param(self.p_param_name, ParamTypes.SPARSE)
        p = p.tocsr()
        r = self.project.load_param(self.r_param_name, ParamTypes.SPARSE)
        r = r.tolil()
        logger.debug('time2 = %.2f' % (time.time() - t0))

        # Iterate on steps. For each step try to activate other nodes.
        i = 0
        while cur_step:
            t0 = time.time()
            i += 1
            logger.debug('step %d ...' % i)

            next_step = []

            for node in cur_step:
                u = node.user_id  # sender user id
                u_i = self.user_map[u]
                p_u = np.squeeze(np.array(p[u_i, :].todense()))  # probabilities of the children of u

                # Iterate on children of u
                for v_i in np.nonzero(p_u)[0]:
                    v = user_ids[int(v_i)]  # receiver (child) user id
                    if v in activated:
                        continue

                    # Set the threshold or sample it randomly if None.
                    if threshold is None:
                        thresh = random.random()
                    else:
                        thresh = threshold

                    # Update probability of activation of the receiver.
                    p_u_v = p_u[v_i]
                    if v not in self.probabilities:
                        self.probabilities[v] = p_u_v
                    else:
                        self.probabilities[v] = max(p_u_v, self.probabilities[v])

                    if thresh <= p_u[v_i]:
                        # Get delay parameter.
                        delay_param = r[u_i, v_i]
                        if delay_param == 0:
                            continue
                        if delay_param < 0:  # Due to some rare bugs in delays
                            delay_param = -delay_param

                        # Sample delay from exponential distribution and calculate the receive time.
                        delay = np.random.exponential(delay_param)  # in days
                        # delay = delay_param  # in days
                        send_dt = str_to_datetime(node.datetime)
                        receive_dt = send_dt + timedelta(days=delay)
                        if receive_dt < now:
                            continue

                        # Add it in the tree.
                        send_dt = str_to_datetime(node.datetime)
                        receive_dt = send_dt + timedelta(days=1)
                        child = CascadeNode(v, datetime=receive_dt.strftime(DT_FORMAT))
                        node.children.append(child)
                        activated.append(v)
                        next_step.append(child)
                        logger.debug('a reshare predicted')
            cur_step = sorted(next_step, key=lambda n: n.datetime)
            logger.debug('time = %.2f' % (time.time() - t0))

        return tree


class ParamTypes:
    JSON = 'json'
    ARRAY = 'array'
    SPARSE = 'sparse'
    SPARSE_LIST = 'splist'
    GRAPH = 'graph'


class GraphTypes:
    RESHARES = 0
    RELATIONS = 1


class Project(object):
    def __init__(self, project_name):
        self.project_name = project_name
        self.project_path = os.path.join(settings.BASEPATH, 'data', project_name)
        # Create the project path if does not exist.
        if not os.path.exists(self.project_path):
            os.mkdir(self.project_path)
        self.training = None
        self.validation = None
        self.test = None
        self.trees = None

    def save_sets(self, train_set, val_set, test_set):
        # Dump the json into the file.
        self.training = list(train_set)
        self.validation = list(val_set)
        self.test = list(test_set)
        data = {'training': train_set, 'validation': val_set, 'test': test_set}
        if not os.path.exists(self.project_path):
            os.mkdir(self.project_path)
        sample_path = os.path.join(self.project_path, 'samples.json')
        json.dump(data, open(sample_path, 'w'), indent=4)

    def load_sets(self):
        sample_set_path = os.path.join(self.project_path, 'samples.json')
        if os.path.exists(sample_set_path):
            data = json.load(open(sample_set_path))
            train_memes, val_memes, test_memes = data['training'], data['validation'], data['test']
            self.training = [ObjectId(_id) for _id in train_memes]
            self.validation = [ObjectId(_id) for _id in val_memes]
            self.test = [ObjectId(_id) for _id in test_memes]
        else:
            raise Exception('Data sample not found. Run sampledata command.')

        return self.training, self.validation, self.test

    def load_trees(self):
        """
        Load trees of memes in training and test sets.
        :return:
        """
        # Load trees from the json file.
        try:
            trees = self.load_param('trees', ParamTypes.JSON)
            trees = {ObjectId(key): value for key, value in trees.items()}
            # Convert tree dictionaries to tree objects.
            logger.debug('converting trees to objects ...')
            trees = {meme_id: CascadeTree().from_dict(tree) for meme_id, tree in trees.items()}
            logger.debug('done')
        except FileNotFoundError:
            try:
                trees_path = os.path.join(settings.BASEPATH, 'data', 'trees.json')
                logger.info('loading trees ...')
                trees = json.load(open(trees_path, 'r'))

                # Keep just trees of the training and test set.
                trees = {ObjectId(key): value for key, value in trees.items()}
                trees = {meme_id: trees[meme_id] for meme_id in self.training + self.test}
                # Save trees for the project.
                self.save_param(trees, 'trees', ParamTypes.JSON)
                # Convert tree dictionaries to tree objects.
                logger.debug('converting trees to objects ...')
                trees = {meme_id: CascadeTree().from_dict(tree) for meme_id, tree in trees.items()}
                logger.debug('done')
            except FileNotFoundError:
                logger.info('trees not found. extracting ...')
                trees = {}
                i = 0
                all_memes = self.training + self.validation + self.test
                count = len(all_memes)
                for meme_id in all_memes:
                    tree = CascadeTree().extract_cascade(meme_id)
                    trees[meme_id] = tree
                    i += 1
                    if i % 10 == 0:
                        logger.info('%d%% done', i * 100 / count)
                trees_dict = {str(meme_id): tree.get_dict() for meme_id, tree in trees.items()}
                self.save_param(trees_dict, 'trees', ParamTypes.JSON)  # Save trees for the project.

        self.trees = trees
        return trees

    def load_or_extract_graph(self, graph_type=GraphTypes.RESHARES):
        """
        Load or extract the graph of memes in training set.
        :return:
        """
        graph_fname = 'graph'
        train_set, _, _ = self.load_sets()

        try:
            graph = self.load_param(graph_fname, ParamTypes.GRAPH)

        except:  # If graph data does not exist.
            post_ids = self.__get_memes_post_ids(train_set)
            if graph_type == GraphTypes.RESHARES:
                graph = self.__extract_reshare_graph(post_ids, train_set, graph_fname)
            else:
                graph = self.__extract_rel_graph(post_ids, graph_fname)

        return graph

    def load_or_extract_act_seq(self):
        """
        Load or extract list of activation sequences of memes in training set.
        :return:
        """
        seq_fname = 'sequences'
        train_set, _, _ = self.load_sets()

        try:
            seq_copy = self.load_param(seq_fname, ParamTypes.JSON)
            sequences = {}
            for m in seq_copy:
                users = [ObjectId(item[0]) for item in seq_copy[m]['cascade']]
                times = [item[1] for item in seq_copy[m]['cascade']]
                sequences[ObjectId(m)] = ActSequence(users=users, times=times, max_t=seq_copy[m]['max_t'])

        except Exception as e:  # If graph data does not exist.
            post_ids = self.__get_memes_post_ids(train_set)
            sequences = self.__extract_act_seq(post_ids, train_set, seq_fname)

        return sequences

    def load_or_extract_graph_seq(self, graph_type=GraphTypes.RESHARES):
        """
        Load the graph, and list of activation sequences of the project if saved before.
        Otherwise extract them from the training set and return them.
        :return: tuple of graph, and list of activation sequences
        """
        graph_fname = 'graph'
        seq_fname = 'sequences'

        train_set, _, _ = self.load_sets()
        post_ids = None

        try:
            graph = self.load_param(graph_fname, ParamTypes.GRAPH)
            graph = relabel_nodes(graph, {n: ObjectId(n) for n in graph.nodes()})

        except:  # If graph does not exist.
            post_ids = self.__get_memes_post_ids(train_set)
            if graph_type == GraphTypes.RESHARES:
                graph = self.__extract_reshare_graph(post_ids, train_set, graph_fname)
            else:
                graph = self.__extract_rel_graph(post_ids, graph_fname)

        try:
            seq_copy = self.load_param(seq_fname, ParamTypes.JSON)
            sequences = {}
            for m in seq_copy:
                users = [ObjectId(item[0]) for item in seq_copy[m]['cascade']]
                times = [item[1] for item in seq_copy[m]['cascade']]
                sequences[ObjectId(m)] = ActSequence(users=users, times=times, max_t=seq_copy[m]['max_t'])

        except:  # If sequence data does not exist.
            if post_ids is None:
                post_ids = self.__get_memes_post_ids(train_set)
            sequences = self.__extract_act_seq(post_ids, train_set, seq_fname)

        return graph, sequences

    @time_measure()
    def __get_memes_post_ids(self, meme_ids):
        logger.info('querying posts ids ...')
        db = DBManager().db
        post_ids = [pm['post_id'] for pm in
                    db.postmemes.find({'meme_id': {'$in': meme_ids}}, {'_id': 0, 'post_id': 1})]
        return post_ids

    def __extract_act_seq(self, posts_ids, meme_ids, seq_fname):
        """
        Extract list of activation sequences from given meme id's.
        :param posts_ids:   list of posts id's
        :param meme_ids:    meme id's
        :param seq_fname:   the file name to save activation sequences data
        :return:            list of ActSequence's
        """
        t0 = time.time()
        logger.info('extracting act. sequences from %d posts ...' % len(posts_ids))
        post_count = len(posts_ids)
        users = {m: [] for m in meme_ids}
        times = {m: [] for m in meme_ids}

        db = DBManager().db

        # Iterate on posts to extract activation sequences.
        i = 0
        while True:
            posts = db.posts.find({'_id': {'$in': posts_ids}}, ['author_id', 'datetime'], no_cursor_timeout=True) \
                .sort('datetime').skip(i)

            try:
                for post in posts:
                    if post['datetime'] is not None:
                        for pm in db.postmemes.find({'post_id': post['_id'], 'meme_id': {'$in': meme_ids}},
                                                    {'_id': 0, 'meme_id': 1}):
                            meme_id = pm['meme_id']
                            if post['author_id'] not in users[meme_id]:
                                users[meme_id].append(post['author_id'])
                                times[meme_id].append(post['datetime'])
                    i += 1
                    if i % (post_count / 10) == 0:
                        logger.info('%d%% posts done' % (i * 100 / post_count))
                posts.close()
                break
            except CursorNotFound:
                logger.info('Lost cursor! retrying with skipping %d first posts ...', i)

        logger.info('setting relative times and max times ...')
        max_t = {}
        i = 0
        for meme in db.memes.find({'_id': {'$in': meme_ids}}, ['last_time', 'first_time']):
            mid = meme['_id']
            times[mid] = [(t - meme['first_time']).total_seconds() / (3600.0 * 24) for t in
                          times[mid]]  # number of days
            max_t[mid] = (meme['last_time'] - meme['first_time']).total_seconds() / (3600.0 * 24)  # number of days
            i += 1
            if i % (len(meme_ids) / 10) == 0:
                logger.info('%d%% done' % (i * 100 / len(meme_ids)))

        sequences = {}
        for m in meme_ids:
            if users[m]:
                sequences[m] = ActSequence(users[m], times[m], max_t[m])

        logger.info('saving act. sequences ...')
        seq_copy = {}
        for m in sequences:
            seq_copy[str(m)] = {
                'cascade': [(str(sequences[m].users[i]), sequences[m].times[i]) for i in
                            range(len(sequences[m].users))],
                'max_t': sequences[m].max_t
            }
        self.save_param(seq_copy, seq_fname, ParamTypes.JSON)
        del seq_copy

        logger.info('act. seq. extraction time: %.2f min' % ((time.time() - t0) / 60.0))
        return sequences

    def __extract_reshares(self, db, post_ids):
        if len(post_ids) < 400000:
            reshares = db.reshares.find(
                {'post_id': {'$in': post_ids}, 'reshared_post_id': {'$in': post_ids}},
                {'_id': 0, 'post_id': 1, 'reshared_post_id': 1, 'user_id': 1, 'ref_user_id': 1}).sort('datetime')
            return list(reshares)
        else:
            reshares = []
            step = 800000 - len(post_ids)
            for i in range(0, len(post_ids), step):
                logger.debug('fetching reshares of post ids from %d to %d ...', i, i + step)
                reshares.extend(list(db.reshares.find(
                    {'post_id': {'$in': post_ids[i: i + step]}, 'reshared_post_id': {'$in': post_ids}},
                    {'_id': 0, 'post_id': 1, 'reshared_post_id': 1, 'user_id': 1, 'ref_user_id': 1, 'datetime': 1})))
                logger.debug('number of reshares: %d', len(reshares))
            reshares.sort(key=lambda resh: resh['datetime'])
            return reshares

    def __extract_reshare_graph(self, post_ids, meme_ids, graph_fname):
        """
        Extract graph from given meme id's.
        :param post_ids:    list of posts id's related to memes
        :param meme_ids:    list of memes id's
        :param graph_fname: the file name to save graph data
        :return:            directed graph of all reshares
        """
        logger.info('extracting graph of reshares ...')
        t0 = time.time()

        logger.info('querying reshares ...')
        db = DBManager().db
        reshares = self.__extract_reshares(db, post_ids)
        resh_count = len(reshares)
        logger.info('time: %.2f min', (time.time() - t0) / 60.0)

        logger.info('extracting graph from %d posts and %d reshares ...', len(post_ids), resh_count)
        edges = []
        meme_ids = set(meme_ids)
        i = 0

        # Iterate on reshares to extract graph edges.
        for resh in reshares:
            user_id = resh['user_id']
            ref_user_id = resh['ref_user_id']
            if user_id != ref_user_id:
                src_meme_ids = {pm['meme_id'] for pm in
                                db.postmemes.find({'post_id': resh['reshared_post_id']}, {'_id': 0, 'meme_id': 1})}
                dest_meme_ids = {pm['meme_id'] for pm in
                                 db.postmemes.find({'post_id': resh['post_id']}, {'_id': 0, 'meme_id': 1})}
                common_memes = meme_ids & src_meme_ids & dest_meme_ids
                if common_memes:
                    edges.append((ref_user_id, user_id))
            i += 1
            if i % (resh_count / 10) == 0:
                logger.info('%d%% reshares done' % (i * 100 / resh_count))

        graph = DiGraph()
        graph.add_edges_from(edges)

        logger.info('saving graph ...')
        self.save_param(graph, graph_fname, ParamTypes.GRAPH)

        logger.info('graph extraction time: %.2f min' % ((time.time() - t0) / 60.0))
        return graph

    def get_all_nodes(self):
        if self.trees is None:
            self.load_trees()

        nodes = set()
        for tree in self.trees.values():
            nodes.update(tree.node_ids())
        return list(nodes)

    SUFFIXES = {
        ParamTypes.JSON: 'json',
        ParamTypes.ARRAY: 'npy',
        ParamTypes.SPARSE: 'npz',
        ParamTypes.SPARSE_LIST: 'npz',
        ParamTypes.GRAPH: 'txt'
    }

    def save_param(self, param, name, type):
        path = os.path.join(self.project_path, '%s.%s' % (name, self.SUFFIXES[type]))
        mkdir_rec(os.path.dirname(path))

        if type == ParamTypes.JSON:
            json.dump(param, open(path, 'w'), indent=1)
        elif type == ParamTypes.ARRAY:
            np.save(path, param)
        elif type == ParamTypes.SPARSE:
            save_sparse(path, param)
        elif type == ParamTypes.SPARSE_LIST:
            save_sparse_list(path, param)
        elif type == ParamTypes.GRAPH:
            write_adjlist(param, path)
        else:
            raise Exception('invalid type "%s"' % type)

    def load_param(self, name, type):
        path = os.path.join(self.project_path, '%s.%s' % (name, self.SUFFIXES[type]))
        if type == ParamTypes.JSON:
            return json.load(open(path))
        elif type == ParamTypes.ARRAY:
            return np.load(path)
        elif type == ParamTypes.SPARSE:
            return load_sparse(path)
        elif type == ParamTypes.SPARSE_LIST:
            return load_sparse_list(path)
        elif type == ParamTypes.GRAPH:
            graph = DiGraph()
            graph = read_adjlist(path, create_using=graph)
            return graph
        else:
            raise Exception('invalid type "%s"' % type)

    def delete_param(self, name, type):
        path = os.path.join(self.project_path, '%s.%s' % (name, self.SUFFIXES[type]))
        os.remove(path)
