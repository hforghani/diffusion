# -*- coding: utf-8 -*-
from datetime import timedelta
import json
import logging
import os
import random
import time

from anytree import Node, RenderTree
from django.conf import settings
from networkx import DiGraph, read_adjlist, relabel_nodes, write_adjlist
import numpy as np

from mongo import mongodb
from utils.numpy_utils import load_sparse, save_sparse, save_sparse_list, load_sparse_list
from utils.time_utils import str_to_datetime, DT_FORMAT

logger = logging.getLogger('cascade.models')


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
        return {'user_id': self.user_id,
                'datetime': self.datetime,
                'post_id': self.post_id,
                'parent_id': self.parent_id,
                'children': [node.get_dict() for node in self.children]}

    def get_detailed_dict(self, user_map):
        """
        Get dictionary of the object. Replace user_id by user details dictionary.
        param user_map: dictionary of user id's to users.
        """
        return {'user': user_map[self.user_id].get_dict(),
                'datetime': self.datetime,
                'post_id': self.post_id,
                'parent_id': self.parent_id,
                'children': [node.get_dict() for node in self.children]}

    def from_dict(self, node_dict):
        """
        Set attributes from dictionary.
        node_dict: dictionary of node,
        parent_id: parent node id,
        users_map: dictionary of mapping from user id's to users.
        """
        self.user_id = node_dict['user_id']
        self.datetime = node_dict['datetime']
        self.post_id = node_dict['post_id']
        self.parent_id = node_dict['parent_id']
        self.children = [CascadeNode().from_dict(node) for node in node_dict['children']]
        return self

    def copy(self):
        """
        Get an independent copy of the object.
        """
        node = CascadeNode(self.user_id, self.datetime, self.post_id, self.parent_id)
        node.children = [child.copy() for child in self.children]
        return node

    def depth(self):
        depth = 0
        for node in self.children:
            depth = max(depth, node.depth() + 1)
        return depth

    def __create_anytree_node(self):
        node = Node('{}({})'.format(self.user_id, self.post_id))
        for child in self.children:
            child_node = child.__create_anytree_node()
            child_node.parent = node
        return node

    def render(self):
        node = self.__create_anytree_node()
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

    def extract_cascade(self, meme_id, log=False):
        t1 = time.time()

        # Fetch posts related to the meme and reshares.
        post_ids = [pm['post_id'] for pm in mongodb.postmemes.find({'meme_id': meme_id}, {'_id': 0, 'post_id': 1})]
        posts = mongodb.posts.find({'_id': {'$in': post_ids}}, {'url': 0}).sort('datetime')

        user_ids = [p['author_id'] for p in posts]
        reshares = mongodb.reshares.find({'post_id': {'$in': post_ids}, 'reshared_post_id': {'$in': post_ids}}) \
            .sort('datetime')
        if log:
            logger.info('\tTREE: time 1 = %.2f' % (time.time() - t1))

        # Create nodes for the users.
        t1 = time.time()
        nodes = {}
        visited = {uid: False for uid in user_ids}  # Set visited True if the node has been visited.
        for user_id in user_ids:
            nodes[user_id] = CascadeNode(user_id)
        if log:
            logger.info('\tTREE: time 2 = %.2f' % (time.time() - t1))

        # Create diffusion edge if a user reshares to another for the first time. Note that reshares are sorted by time.
        t1 = time.time()
        if log:
            logger.info('\tTREE: reshares count = %d' % reshares.count())
        self.roots = []
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
                self.roots.append(parent)

            if not visited[child_id]:  # Any other node
                child = nodes[child_id]
                parent.children.append(child)
                child.parent_id = parent_id
                child.post_id = reshare['post_id']
                child.datetime = reshare['datetime'].strftime(DT_FORMAT)
                visited[child_id] = True
        if log:
            logger.info('\tTREE: time 3 = %.2f' % (time.time() - t1))

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
                self.roots.append(node)
        if log:
            logger.info('\tTREE: time 4 = %.2f' % (time.time() - t1))

        # Calculate tree depth.
        self.depth = self.__calc_depth()

        return self

    def get_dict(self):
        return [node.get_dict() for node in self.roots]

    def get_detailed_dict(self):
        user_ids = self.node_ids()
        users = UserAccount.objects.filter(id__in=user_ids)
        user_map = {user.id: user for user in users}
        return [node.get_detailed_dict(user_map) for node in self.roots]

    def from_dict(self, tree_dict):
        self.roots = []
        for node in tree_dict:
            self.roots.append(CascadeNode().from_dict(node))
        return self

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
        if node is None:
            node = self.roots
        leaves = []
        if not node.children:
            leaves.append(node)
        else:
            for child in node.children:
                leaves.extend(self.get_leaves(child))
        return leaves

    def node_ids(self):
        return [node.user_id for node in self.nodes()]

    def nodes(self, node=None):
        nodes_list = []
        if node is None:
            for node in self.roots:
                nodes_list.extend(self.nodes(node))
        else:
            nodes_list = [node]
            for child in node.children:
                nodes_list.extend(self.nodes(child))
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

    def copy(self):
        tree_copy = [root.copy() for root in self.roots]
        return CascadeTree(tree_copy)

    def __calc_depth(self):
        depth = 0
        for node in self.roots:
            depth = max(depth, node.depth())
        return depth

    def render(self):
        return '\n'.join([root.render() for root in self.roots])


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
        self.max_delay = 1000
        self.probabilities = {}  # dictionary of node id's to probabilities of activation
        self.user_ids = None
        self.users_map = None

        self.w = self.project.load_param(self.w_param_name, ParamTypes.SPARSE)
        self.w = self.w.tocsr()
        self.r = self.project.load_param(self.r_param_name, ParamTypes.ARRAY)

    def fit(self):
        pass

    # @profile
    def predict(self, initial_tree, threshold=None, user_ids=None, users_map=None, log=False):
        """
        Predict activation cascade in the future starting from initial nodes in self.tree.
        Set the final tree again in self.tree.
        :param initial_tree:    Initial tree of activated nodes
        :param threshold:       Threshold of activation probability. IF None, threshold is sampled randomly.
        :param user_ids:        List of possible users for activation. All of users if value is None.
        :param log:             Log in console if True else does not log.
        :return:                Returns self.tree
        """
        if not isinstance(initial_tree, CascadeTree):
            raise ValueError('tree must be a CascadeTree')
        tree = initial_tree.copy()

        # Initialize values.
        t0 = time.time()
        cur_step = sorted(tree.nodes(), key=lambda n: n.datetime)  # Set tree nodes as initial step.
        activated = tree.nodes()
        self.probabilities = {}

        if user_ids is None or users_map is None:
            user_ids = UserAccount.objects.values_list('id', flat=True).order_by('id')
            users_map = {user_ids[i]: i for i in range(len(user_ids))}
        self.user_ids = user_ids
        self.users_map = users_map

        thresholds = {}
        if log:
            logger.info('time1 = %.2f' % (time.time() - t0))

        # Get weights and delay vectors.
        t0 = time.time()
        if log:
            logger.info('time2 = %.2f' % (time.time() - t0))

        # Iterate on steps. For each step try to activate other nodes.
        step = 0
        while cur_step:
            t0 = time.time()
            step += 1
            # if log:
            #     logger.info('step %d ...' % step)

            next_step = []

            # Iterate on current step nodes to check if a child will be activated.
            for node in cur_step:
                u = node.user_id  # sender user id
                u_i = self.users_map[u]
                # w_u = np.squeeze(np.array(w[u_i, :].todense()))  # weights of the children of u
                w_u = self.w[u_i, :]

                # Iterate on children of u
                # for v_i in np.nonzero(w_u)[0]:
                for i in range(w_u.nnz):
                    v_i = w_u.indices[i]
                    v_i = int(v_i)
                    v = user_ids[v_i]  # receiver (child) user id
                    if v in activated:
                        continue
                    if v not in self.probabilities:
                        self.probabilities[v] = 0
                        # self.probabilities[v] += w_u[v_i]
                    self.probabilities[v] += w_u.data[i]

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
                        # if delay_param > 0:
                        #     delay = np.random.exponential(delay_param)  # in days
                        # else:
                        #     if log:
                        #         logger.warn('delay param = {}'.format(delay_param))
                        #     delay = 1000    # a very large delay!
                        send_dt = str_to_datetime(node.datetime)
                        receive_dt = send_dt + timedelta(days=delay)

                        # Add it to the tree.
                        child = CascadeNode(v, datetime=receive_dt.strftime(DT_FORMAT))
                        # child = CascadeNode(v)
                        node.children.append(child)
                        activated.append(v)
                        next_step.append(child)
                        # if log:
                        #     logger.info('a reshare predicted')
            cur_step = sorted(next_step, key=lambda n: n.datetime)
            if log:
                logger.info('step %d, time = %.2f' % (step, time.time() - t0))

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

    def predict(self, initial_tree, threshold=None, user_ids=None, log=False):
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
                user_ids = UserAccount.objects.values_list('id', flat=True).order_by('id')
            self.user_map = {user_ids[i]: i for i in range(len(user_ids))}
        if log:
            logger.info('time1 = %.2f' % (time.time() - t0))

        # Get diffusion probabilities and delay vectors.
        t0 = time.time()
        p = self.project.load_param(self.p_param_name, ParamTypes.SPARSE)
        p = p.tocsr()
        r = self.project.load_param(self.r_param_name, ParamTypes.SPARSE)
        r = r.tolil()
        if log:
            logger.info('time2 = %.2f' % (time.time() - t0))

        # Iterate on steps. For each step try to activate other nodes.
        i = 0
        while cur_step:
            t0 = time.time()
            i += 1
            if log:
                logger.info('\tstep %d ...' % i)

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
                        if log:
                            logger.info('\ta reshare predicted')
            cur_step = sorted(next_step, key=lambda n: n.datetime)
            if log:
                logger.info('time = %.2f' % (time.time() - t0))

        return tree


class ParamTypes(object):
    JSON = 'json'
    ARRAY = 'array'
    SPARSE = 'sparse'
    SPARSE_LIST = 'splist'
    GRAPH = 'graph'


class Project(object):
    def __init__(self, project_name):
        self.project_name = project_name
        self.project_path = os.path.join(settings.BASEPATH, 'data', project_name)
        # Create the project path if does not exist.
        if not os.path.exists(self.project_path):
            os.mkdir(self.project_path)
        self.training = None
        self.test = None
        self.trees = None

    def save_data(self, test_set, train_set):
        # Dump the json into the file.
        self.training = list(train_set)
        self.test = list(test_set)
        data = {'training': train_set, 'test': test_set}
        if not os.path.exists(self.project_path):
            os.mkdir(self.project_path)
        sample_path = os.path.join(self.project_path, 'samples.json')
        json.dump(data, open(sample_path, 'w'), indent=4)

    def load_train_test(self):
        sample_set_path = os.path.join(self.project_path, 'samples.json')
        if os.path.exists(sample_set_path):
            data = json.load(open(sample_set_path))
            train_memes, test_memes = data['training'], data['test']
        else:
            raise Exception('Data sample not found. Run sampledata command.')

        self.training = train_memes
        self.test = test_memes

        return train_memes, test_memes

    def load_trees(self, verbosity=settings.VERBOSITY):
        """
        Load trees of memes in training and test sets.
        :return:
        """
        # Load trees from the json file.
        try:
            trees = self.load_param('trees', ParamTypes.JSON)
            trees = {int(key): value for key, value in trees.items()}
            # Convert tree dictionaries to tree objects.
            if verbosity:
                logger.info('converting trees to objects ...')
            trees = {meme_id: CascadeTree().from_dict(tree) for meme_id, tree in trees.items()}
        except FileNotFoundError:
            try:
                trees_path = os.path.join(settings.BASEPATH, 'data', 'trees.json')
                logger.info('loading trees ...')
                trees = json.load(open(trees_path, 'r'))

                # Keep just trees of the training and test set.
                trees = {int(key): value for key, value in trees.items()}
                trees = {meme_id: trees[meme_id] for meme_id in self.training + self.test}
                # Save trees for the project.
                self.save_param(trees, 'trees', ParamTypes.JSON)
                # Convert tree dictionaries to tree objects.
                if verbosity:
                    logger.info('converting trees to objects ...')
                trees = {meme_id: CascadeTree().from_dict(tree) for meme_id, tree in trees.items()}
            except FileNotFoundError:
                trees = {}
                for meme_id in self.training + self.test:
                    tree = CascadeTree().extract_cascade(meme_id)
                    trees[meme_id] = tree
                    trees_dict = {meme_id: tree.get_dict() for meme_id, tree in trees.items()}
                    # Save trees for the project.
                    self.save_param(trees_dict, 'trees', ParamTypes.JSON)

        self.trees = trees
        return trees

    def load_or_extract_graph(self):
        """
        Load or extract the graph of memes in training set.
        :return:
        """
        graph_fname = 'graph'
        train_set, test_set = self.load_train_test()

        try:
            graph = self.load_param(graph_fname, ParamTypes.GRAPH)

        except:  # If graph data does not exist.
            logger.info('\tquerying posts and reshares ...')
            t0 = time.time()
            posts = Post.objects.filter(postmeme__meme_id__in=train_set).distinct().order_by('datetime')
            reshares = Reshare.objects.filter(post__in=posts, reshared_post__in=posts).distinct().order_by('datetime')
            resh_count = reshares.count()
            post_count = posts.count()
            logger.info('\ttime: %.2f min' % ((time.time() - t0) / 60.0))

            # Create graph.
            logger.info('\textracting graph from %d posts and %d reshares ...' % (post_count, resh_count))
            meme_ids = Meme.objects.order_by('id').values_list('id', flat=True)
            graph = self.__extract_graph(reshares, meme_ids)

            logger.info('\tsaving data ...')
            self.save_param(graph, graph_fname, ParamTypes.GRAPH)

        return graph

    def load_or_extract_act_seq(self):
        """
        Load or extract list of activation sequences of memes in training set.
        :return:
        """
        seq_fname = 'sequences'
        train_set, test_set = self.load_train_test()

        try:
            seq_copy = self.load_param(seq_fname, ParamTypes.JSON)
            sequences = {}
            for m in seq_copy:
                users = [item[0] for item in seq_copy[m]['cascade']]
                times = [item[1] for item in seq_copy[m]['cascade']]
                sequences[int(m)] = ActSequence(users=users, times=times, max_t=seq_copy[m]['max_t'])

        except:  # If graph data does not exist.
            logger.info('\tquerying posts and reshares ...')
            t0 = time.time()
            posts = Post.objects.filter(postmeme__meme_id__in=train_set).distinct().order_by('datetime')
            post_count = posts.count()
            logger.info('\ttime: %.2f min' % ((time.time() - t0) / 60.0))

            # Create dictionary of first times of the memes.
            logger.info('\textracting first times ...')
            first_times = Meme.objects.values('id', 'first_time')
            first_times = {obj['id']: obj['first_time'] for obj in first_times}

            # Create graph and cascade data.
            logger.info('\textracting act sequences from %d posts ...' % post_count)
            meme_ids = Meme.objects.order_by('id').values_list('id', flat=True)
            sequences = self.__extract_act_seq(posts, first_times, meme_ids)

            logger.info('\tsetting max times ...')
            i = 0
            for meme in Meme.objects.filter(id__in=sequences.keys()).iterator():
                sequences[meme.id].max_t = (meme.last_time - meme.first_time).total_seconds() / (
                    3600.0 * 24)  # number of days
                i += 1
                if i % (len(meme_ids) / 10) == 0:
                    logger.info('\t\t%d%% done' % (i * 100 / len(meme_ids)))

            logger.info('\tsaving data ...')
            seq_copy = {}
            for m in sequences:
                seq_copy[m] = {
                    'cascade': [(sequences[m].users[i], sequences[m].times[i]) for i in range(len(sequences[m].users))],
                    'max_t': sequences[m].max_t
                }
            self.save_param(seq_copy, seq_fname, ParamTypes.JSON)

        return sequences

    def load_or_extract_graph_seq(self):
        """
        Load the graph, and list of activation sequences of the project if saved before.
        Otherwise extract them from the training set and return them.
        :return: tuple of graph, and list of activation sequences
        """
        graph_fname = 'graph'
        seq_fname = 'sequences'

        train_set, test_set = self.load_train_test()

        try:
            graph = self.load_param(graph_fname, ParamTypes.GRAPH)
            seq_copy = self.load_param(seq_fname, ParamTypes.JSON)
            sequences = {}
            for m in seq_copy:
                users = [item[0] for item in seq_copy[m]['cascade']]
                times = [item[1] for item in seq_copy[m]['cascade']]
                sequences[int(m)] = ActSequence(users=users, times=times, max_t=seq_copy[m]['max_t'])

        except:  # If graph and sequence data does not exist.
            logger.info('\tquerying posts and reshares ...')
            t0 = time.time()
            posts = Post.objects.filter(postmeme__meme_id__in=train_set).distinct().order_by('datetime')
            reshares = Reshare.objects.filter(post__in=posts, reshared_post__in=posts).distinct().order_by('datetime')
            resh_count = reshares.count()
            post_count = posts.count()
            logger.info('\ttime: %.2f min' % ((time.time() - t0) / 60.0))

            # Create dictionary of first times of the memes.
            logger.info('\textracting first times ...')
            first_times = Meme.objects.values('id', 'first_time')
            first_times = {obj['id']: obj['first_time'] for obj in first_times}

            # Create graph and activation sequence.
            logger.info('\textracting cascades from %d posts and %d reshares ...' % (post_count, resh_count))
            meme_ids = Meme.objects.order_by('id').values_list('id', flat=True)
            graph = self.__extract_graph(reshares, meme_ids)
            sequences = self.__extract_act_seq(posts, first_times, meme_ids)

            logger.info('\tsetting max times ...')
            i = 0
            for meme in Meme.objects.filter(id__in=sequences.keys()).iterator():
                sequences[meme.id].max_t = (meme.last_time - meme.first_time).total_seconds() / (
                    3600.0 * 24)  # number of days
                i += 1
                if i % (len(meme_ids) / 10) == 0:
                    logger.info('\t\t%d%% done' % (i * 100 / len(meme_ids)))

            logger.info('\tsaving data ...')
            seq_copy = {}
            for m in sequences:
                seq_copy[m] = {
                    'cascade': [(sequences[m].users[i], sequences[m].times[i]) for i in range(len(sequences[m].users))],
                    'max_t': sequences[m].max_t
                }
            self.save_param(seq_copy, seq_fname, ParamTypes.JSON)
            del seq_copy
            self.save_param(graph, graph_fname, ParamTypes.GRAPH)

        return graph, sequences

    def __extract_graph(self, reshares, meme_ids):
        """
        Extract graph from given meme id's.
        :param reshares:    queryset of reshares related to posts of given meme id's
        :param meme_ids:    meme id's
        :return:            directed graph of all reshares
        """
        t0 = time.time()
        edges = []
        meme_ids = set(meme_ids)
        resh_count = reshares.count()
        i = 0

        # Iterate on reshares to extract graph edges.
        for resh in reshares.all():
            # user_id = resh.user_id
            user_id = resh.post.author_id
            # ref_user_id = resh.ref_user_id
            ref_user_id = resh.reshared_post.author_id
            if user_id != ref_user_id:
                common_memes = meme_ids & set(resh.reshared_post.postmeme_set.values_list('meme_id', flat=True)) & set(
                    resh.post.postmeme_set.values_list('meme_id', flat=True))
                if common_memes:
                    edges.append((ref_user_id, user_id))
            i += 1
            if i % (resh_count / 10) == 0:
                logger.info('\t\t%d%% reshares done' % (i * 100 / resh_count))

        graph = DiGraph()
        graph.add_edges_from(edges)

        logger.info('\t\tgraph extraction time: %.2f min' % ((time.time() - t0) / 60.0))
        return graph

    def __extract_act_seq(self, posts, first_times, meme_ids):
        """
        Extract list of activation sequences from given meme id's.
        :param posts:       queryset of posts
        :param first_times: dictionary of meme id's to their first post time
        :param meme_ids:    meme id's
        :return:            list of ActSequence's
        """
        t0 = time.time()
        meme_ids = set(meme_ids)
        post_count = posts.count()
        users = {m: [] for m in meme_ids}
        times = {m: [] for m in meme_ids}
        i = 0

        # Iterate on posts to extract activation sequences.
        for post in posts.all():
            for m in post.postmeme_set.values_list('meme_id', flat=True):
                if post.author_id not in users[m]:
                    users[m].append(post.author_id)
                    act_time = (post.datetime - first_times[m]).total_seconds() / (3600.0 * 24)  # number of days
                    times[m].append(act_time)
            i += 1
            if i % (post_count / 10) == 0:
                logger.info('\t\t%d%% posts done' % (i * 100 / post_count))

        del posts
        data = {}
        for m in meme_ids:
            if users[m]:
                data[m] = ActSequence(users[m], times[m])
        logger.info('\t\tact. seq. extraction time: %.2f min' % ((time.time() - t0) / 60.0))
        return data

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
        if type == ParamTypes.JSON:
            json.dump(param, open(path, 'w'), indent=4)
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
            graph = relabel_nodes(graph, {n: int(n) for n in graph.nodes()})
            return graph
        else:
            raise Exception('invalid type "%s"' % type)

    def delete_param(self, name, type):
        path = os.path.join(self.project_path, '%s.%s' % (name, self.SUFFIXES[type]))
        os.remove(path)
