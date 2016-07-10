# -*- coding: utf-8 -*-
from datetime import timedelta
import logging
import random
import time
import numpy as np
from crud.models import UserAccount, Post, Reshare
from utils.numpy_utils import load_sparse
from utils.time_utils import str_to_datetime, DT_FORMAT

logger = logging.getLogger('diffusion.diffusion.models')


class CascadeNode(object):
    def __init__(self, user=None, datetime=None, post_id=None, parent_id=None):
        self.user = user
        self.datetime = datetime
        self.post_id = post_id
        self.parent_id = parent_id
        self.children = []

    def get_dict(self):
        return {'user': self.user.get_dict(),
                'datetime': self.datetime,
                'post_id': self.post_id,
                'parent_id': self.parent_id,
                'children': [node.get_dict() for node in self.children]}

    def from_dict(self, node_dict, users_map):
        """
        Set attributes from dictionary.
        node_dict: dictionary of node,
        parent_id: parent node id,
        users_map: dictionary of mapping from user id's to users.
        """
        self.user = users_map[node_dict['user']['id']]
        self.datetime = node_dict['datetime']
        self.post_id = node_dict['post_id']
        self.parent_id = node_dict['parent_id']
        self.children = [CascadeNode().from_dict(node, users_map) for node in node_dict['children']]
        return self

    def copy(self):
        node = CascadeNode(self.user, self.datetime, self.post_id, self.parent_id)
        node.children = [child.copy() for child in self.children]
        return node


class CascadeTree(object):
    tree = []

    def __init__(self, tree=None):
        if tree is not None:
            if not isinstance(tree, list):
                raise ValueError('tree must be a list of root nodes')
            self.tree = tree

    def extract_cascade(self, meme, log=False):
        # Fetch posts related to the meme and reshares.
        t1 = time.time()
        posts = Post.objects.filter(postmeme__meme=meme).distinct().order_by('datetime')
        user_ids = posts.values_list('author__id', flat=True).distinct()
        reshares = Reshare.objects.filter(post__in=posts, reshared_post__in=posts).distinct().order_by('datetime')
        if log:
            logger.info('\tTREE: time 1 = %.2f' % (time.time() - t1))

        # Create nodes for the users.
        t1 = time.time()
        nodes = {}
        visited = {uid: False for uid in user_ids}  # Set visited True if the node has been visited.
        for user in UserAccount.objects.filter(id__in=user_ids):
            nodes[user.id] = CascadeNode(user)
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
        return self

    def get_dict(self):
        return [node.get_dict() for node in self.tree]

    def from_dict(self, tree_dict):
        user_ids = []
        for node in tree_dict:
            user_ids.extend(self._tree_dict_user_ids(node))
        users = UserAccount.objects.filter(id__in=user_ids)
        users = {user.id: user for user in users}

        self.tree = []
        for node in tree_dict:
            self.tree.append(CascadeNode().from_dict(node, users))
        return self

    def _tree_dict_user_ids(self, node_dict):
        ids = [node_dict['user']['id']]
        for node in node_dict['children']:
            ids.extend(self._tree_dict_user_ids(node))
        return ids

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
        return [node.user.id for node in self.nodes()]

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
            parent_id = node.user.id
            edges_list = [(parent_id, child.user.id) for child in node.children]
            for child in node.childrens:
                edges_list.extend(self.edges(child))
        return edges_list

    def copy(self):
        tree_copy = [root.copy() for root in self.tree]
        return CascadeTree(tree_copy)


class AsLT(object):
    def __init__(self):
        self.max_delay = 999999999
        self.save_paths = {}  # implement in children.
        self.weight_sum = {}  # dictionary of node id's to sums of parents weights

    def fit(self, tree):
        if not isinstance(tree, CascadeTree):
            raise ValueError('tree must be CascadeTree')
        self.tree = tree.copy()
        return self

    def predict(self, log=False):
        if not self.tree:
            raise ValueError('no tree set')

        # Initialize values.
        now = self.tree.max_datetime()  # Find the datetime of now.
        cur_step = sorted(self.tree.nodes(), key=lambda n: n.datetime)  # Set tree nodes as initial step.
        activated = self.tree.nodes()
        self.weight_sum = {}
        user_ids = UserAccount.objects.values_list('id', flat=True).order_by('id')
        user_map = {user_ids[i]: i for i in range(len(user_ids))}

        # Get weights and delay vectors.
        w = load_sparse(self.save_paths['w'])
        r = np.load(self.save_paths['r'])

        # Iterate on steps. For each step try to activate other nodes.
        i = 0
        while cur_step:
            i += 1
            if log:
                logger.info('\tstep %d ...' % i)

            next_step = []

            for node in cur_step:
                u = node.user.id  # sender user id
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
                        child = CascadeNode(UserAccount.objects.get(id=v), datetime=receive_dt.strftime(DT_FORMAT))
                        node.children.append(child)
                        activated.append(v)
                        next_step.append(child)
                        if log:
                            logger.info('\ta reshare predicted')
            cur_step = sorted(next_step, key=lambda n: n.datetime)

        return self.tree


class CascadeData(object):
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
