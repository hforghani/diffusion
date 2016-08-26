# -*- coding: utf-8 -*-
from datetime import timedelta
import json
import logging
import os
import random
import time
from django.conf import settings
from networkx import DiGraph, read_adjlist, relabel_nodes, write_adjlist
import numpy as np
from crud.models import UserAccount, Post, Reshare
from utils.numpy_utils import load_sparse, save_sparse, save_sparse_list, load_sparse_list
from utils.time_utils import str_to_datetime, DT_FORMAT

logger = logging.getLogger('diffusion.diffusion.models')


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


class CascadeTree(object):
    tree = []
    depth = 0

    def __init__(self, tree=None):
        if tree is not None:
            if not isinstance(tree, list):
                raise ValueError('tree must be a list of root nodes')
            self.tree = tree
            self.depth = self._calc_depth()

    def extract_cascade(self, meme_id, log=False):
        # Fetch posts related to the meme and reshares.
        t1 = time.time()
        posts = Post.objects.filter(postmeme__meme=meme_id).distinct().order_by('datetime')
        user_ids = posts.values_list('author__id', flat=True).distinct()
        reshares = Reshare.objects.filter(post__in=posts, reshared_post__in=posts).distinct().order_by('datetime')
        if log:
            logger.info('\tTREE: time 1 = %.2f' % (time.time() - t1))

        # Create nodes for the users.
        t1 = time.time()
        nodes = {}
        visited = {uid: False for uid in user_ids}  # Set visited True if the node has been visited.
        for user in UserAccount.objects.filter(id__in=user_ids):
            nodes[user.id] = CascadeNode(user.id)
        if log:
            logger.info('\tTREE: time 2 = %.2f' % (time.time() - t1))

        # Create diffusion edge if a user reshares to another for the first time. Note that reshares are sorted by time.
        t1 = time.time()
        if log:
            logger.info('\tTREE: reshares count = %d' % reshares.count())
        self.tree = []
        for reshare in reshares:
            child_id = reshare.user_id
            parent_id = reshare.ref_user_id
            if child_id == parent_id:
                continue  # Continue if the reshare is between same users.
            parent = nodes[parent_id]

            if not visited[parent_id]:  # It is a root
                parent.post_id = reshare.reshared_post_id
                parent.datetime = reshare.ref_datetime.strftime(DT_FORMAT)
                visited[parent_id] = True
                self.tree.append(parent)

            if not visited[child_id]:  # Any other node
                child = nodes[child_id]
                parent.children.append(child)
                child.parent_id = parent_id
                child.post_id = reshare.post_id
                child.datetime = reshare.datetime.strftime(DT_FORMAT)
                visited[child_id] = True
        if log:
            logger.info('\tTREE: time 3 = %.2f' % (time.time() - t1))

        # Add users with no diffusion edges as single nodes.
        t1 = time.time()
        first_posts = {}
        for post in posts:
            if post.author_id not in first_posts:
                first_posts[post.author_id] = post
        for uid, node in nodes.items():
            if not visited[uid]:
                post = first_posts[uid]
                node.datetime = post.datetime.strftime(DT_FORMAT)
                node.post_id = post.id
                self.tree.append(node)
        if log:
            logger.info('\tTREE: time 4 = %.2f' % (time.time() - t1))

        # Calculate tree depth.
        self.depth = self._calc_depth()

        return self

    def get_dict(self):
        return [node.get_dict() for node in self.tree]

    def get_detailed_dict(self):
        user_ids = self.node_ids()
        users = UserAccount.objects.filter(id__in=user_ids)
        user_map = {user.id: user for user in users}
        return [node.get_detailed_dict(user_map) for node in self.tree]

    def from_dict(self, tree_dict):
        self.tree = []
        for node in tree_dict:
            self.tree.append(CascadeNode().from_dict(node))
        return self

    def max_datetime(self, node=None):
        """
        Get maximum datetime of nodes.
        """
        if node is None:
            max_dt = None
            if self.tree and self.tree[0].datetime is not None:
                max_dt = str_to_datetime(self.tree[0].datetime)
            for root in self.tree:
                max_dt = max(max_dt, self.max_datetime(root))
        else:
            max_dt = str_to_datetime(node.datetime) if node.datetime is not None else None
            for child in node.children:
                max_dt = max(max_dt, self.max_datetime(child))
        return max_dt

    def get_leaves(self, node=None):
        if node is None:
            node = self.tree
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
            for node in self.tree:
                nodes_list.extend(self.nodes(node))
        else:
            nodes_list = [node]
            for child in node.children:
                nodes_list.extend(self.nodes(child))
        return nodes_list

    def edges(self, node=None):
        edges_list = []
        if node is None:
            for root in self.tree:
                edges_list.extend(self.edges(root))
        else:
            parent_id = node.user_id
            edges_list = [(parent_id, child.user_id) for child in node.children]
            for child in node.children:
                edges_list.extend(self.edges(child))
        return edges_list

    def copy(self):
        tree_copy = [root.copy() for root in self.tree]
        return CascadeTree(tree_copy)

    def _calc_depth(self):
        depth = 0
        for node in self.tree:
            depth = max(depth, node.depth())
        return depth


class AsLT(object):
    def __init__(self, project):
        self.project = project
        self.max_delay = 999999999
        self.save_paths = {'w': '', 'r': ''}  # implement in children.
        self.weight_sum = {}  # dictionary of node id's to sums of parents weights

    def fit(self, tree):
        if not isinstance(tree, CascadeTree):
            raise ValueError('tree must be CascadeTree')
        self.tree = tree.copy()
        return self

    def predict(self, user_ids=None, log=False):
        if not self.tree:
            raise ValueError('no tree set')

        # Initialize values.
        t0 = time.time()
        now = self.tree.max_datetime()  # Find the datetime of now.
        cur_step = sorted(self.tree.nodes(), key=lambda n: n.datetime)  # Set tree nodes as initial step.
        activated = self.tree.nodes()
        self.weight_sum = {}
        if user_ids is None:
            user_ids = UserAccount.objects.values_list('id', flat=True).order_by('id')
        user_map = {user_ids[i]: i for i in range(len(user_ids))}
        if log:
            logger.info('time1 = %.2f' % (time.time() - t0))

        # Get weights and delay vectors.
        t0 = time.time()
        w = self.project.load_param('w', 'sparse')
        #w = load_sparse(self.save_paths['w'])
        w = w.tocsr()
        r = self.project.load_param('r', 'array')
        #r = np.load(self.save_paths['r'])
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
                u_i = user_map[u]
                w_u = np.squeeze(np.array(w[u_i, :].todense()))  # weights of the children of u

                # Iterate on children of u
                for v_i in np.nonzero(w_u)[0]:
                    v = user_ids[int(v_i)]  # receiver (child) user id
                    if v in activated:
                        continue
                    if v not in self.weight_sum:
                        self.weight_sum[v] = 0
                    self.weight_sum[v] += w_u[v_i]

                    # Try to activate the children.
                    sample = random.random()
                    #sample = 0.5
                    #If activated successfully add it the tree and estimate the delay.
                    if sample <= self.weight_sum[v]:
                        # Get delay parameter.
                        delay_param = r[v_i]
                        if delay_param == 0:
                            continue
                        if delay_param < 0:  # Due to some rare bugs in delays
                            delay_param = -delay_param

                        # Sample delay from exponential distribution and calculate the receive time.
                        delay = np.random.exponential(delay_param)  # in days
                        #delay = delay_param  # in days
                        send_dt = str_to_datetime(node.datetime)
                        receive_dt = send_dt + timedelta(days=delay)
                        if receive_dt < now:
                            continue
                        child = CascadeNode(v, datetime=receive_dt.strftime(DT_FORMAT))
                        node.children.append(child)
                        activated.append(v)
                        next_step.append(child)
                        if log:
                            logger.info('\ta reshare predicted')
            cur_step = sorted(next_step, key=lambda n: n.datetime)
            if log:
                logger.info('time = %.2f' % (time.time() - t0))

        return self.tree


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
        data = {'training': train_set, 'test': test_set}
        if not os.path.exists(self.project_path):
            os.mkdir(self.project_path)
        sample_path = os.path.join(self.project_path, 'samples.json')
        json.dump(data, open(sample_path, 'w'), indent=4)

    def load_data(self):
        sample_set_path = os.path.join(self.project_path, 'samples.json')
        if os.path.exists(sample_set_path):
            logger.info('loading training memes ...')
            data = json.load(open(sample_set_path))
            train_memes, test_memes = data['training'], data['test']
        else:
            raise Exception('Data sample not found. Run sampledata command.')

        self.training = train_memes
        self.test = test_memes

        return train_memes, test_memes

    def load_trees(self):
        # Load trees from the json file.
        trees_path = os.path.join(settings.BASEPATH, 'data', 'trees.json')
        if os.path.exists(trees_path):
            logger.info('loading trees ...')
            trees = json.load(open(trees_path, 'r'))
            trees = {long(key): value for key, value in trees.items()}
        else:
            raise Exception('Trees data not found. Run extracttrees command.')

        # Keep just trees of the training and test set.
        trees = {meme_id: trees[meme_id] for meme_id in self.training + self.test}

        # Convert tree dictionaries to tree objects.
        logger.info('converting trees to objects ...')
        trees = {meme_id: CascadeTree().from_dict(tree) for meme_id, tree in trees.items()}

        self.trees = trees
        return trees

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
