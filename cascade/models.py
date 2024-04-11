import json
import os
from datetime import datetime
from functools import reduce

from anytree import Node, RenderTree
from bson.objectid import ObjectId
from networkx import DiGraph, read_adjlist, relabel_nodes, write_adjlist
import numpy as np

import settings
from db.managers import DBManager
from settings import logger
from utils.numpy_utils import load_sparse, save_sparse, save_sparse_list, load_sparse_list
from utils.os_utils import mkdir_rec
from utils.time_utils import str_to_datetime, DT_FORMAT, Timer, time_measure


class CascadeNode:
    def __init__(self, user_id=None, datetime=None, parent_id=None):
        self.user_id = user_id
        self.datetime = datetime
        self.parent_id = parent_id
        self.children = []
        self.probability = None

    def __repr__(self):
        return f'CascadeNode({self.user_id})'

    def __hash__(self):
        return hash(self.user_id)

    def __eq__(self, other):
        return self.user_id == other.user_id

    def to_json(self):
        """
        Get dictionary of the object.
        """
        return {'user_id': str(self.user_id),
                'datetime': self.datetime.strftime(DT_FORMAT),
                'parent_id': str(self.parent_id) if self.parent_id is not None else None,
                'children': [node.to_json() for node in self.children]}

    def from_json(self, node_dict):
        """
        Set attributes from dictionary.
        node_dict: dictionary of node,
        parent_id: parent node id,
        users_map: dictionary of mapping from user id's to users.
        """
        self.user_id = ObjectId(node_dict['user_id'])
        self.datetime = datetime.strptime(node_dict['datetime'], DT_FORMAT)
        self.parent_id = ObjectId(node_dict['parent_id']) if node_dict['parent_id'] is not None else None
        self.children = [CascadeNode().from_json(node) for node in node_dict['children']]
        return self

    def copy(self, max_depth=None):
        """
        Get an independent copy of the object.
        """
        node = CascadeNode(self.user_id, self.datetime, self.parent_id)
        if max_depth is None or max_depth > 0:
            max_depth = max_depth - 1 if max_depth is not None else None
            node.children = [child.copy(max_depth) for child in self.children]
        return node

    def height(self):
        return max([0] + [node.height() + 1 for node in self.children])

    def __create_anytree_node(self):
        node = Node(str(self.user_id))
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


class CascadeTree:
    def __init__(self, roots=None):
        if roots is not None:
            if not isinstance(roots, list):
                raise ValueError('tree must be a list of root nodes')
            self.roots = roots
            self.depth = self.__calc_depth()
            self._id_to_node = {node.user_id: node for node in self.nodes()}
        else:
            self.roots = []
            self.depth = 0
            self._id_to_node = {}

    def to_json(self):
        return [node.to_json() for node in self.roots]

    @classmethod
    def from_json(cls, tree_dict):
        roots = []
        for node in tree_dict:
            roots.append(CascadeNode().from_json(node))
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

    def size(self, node=None):
        if node is None:
            return sum(self.size(node) for node in self.roots)
        else:
            return 1 + sum(self.size(child) for child in node.children)

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

    def nodes_at_depth(self, depth):
        cur_nodes = self.roots.copy()
        for cur_depth in range(depth):
            cur_nodes = reduce(lambda x, y: x + y, [node.children for node in cur_nodes], [])
            if not cur_nodes:
                break
        return cur_nodes

    def depth_of(self, user_id):
        node = self._id_to_node[user_id]
        depth = 0
        while node.parent_id is not None:
            node = self._id_to_node[node.parent_id]
            depth += 1
        return depth

    def edges(self, node=None, max_depth=None):
        nodes = self.nodes(node, max_depth)
        edges_list = [(node.parent_id, node.user_id) for node in nodes if node.parent_id]
        return edges_list

    def copy(self, max_depth=None):
        tree_copy = [root.copy(max_depth) for root in self.roots]
        return CascadeTree(tree_copy)

    def __calc_depth(self):
        depth = 0
        for node in self.roots:
            depth = max(depth, node.height())
        return depth

    def render(self, digest=False):
        return '\n'.join([root.render() for root in self.roots])

    def get_node(self, user_id: ObjectId) -> CascadeNode:
        return self._id_to_node.get(user_id, None)

    def add_node(self, node_id: ObjectId, act_time: datetime = None, parent_id: ObjectId = None) -> CascadeNode:
        """
        Add the child node to the tree. If the parent is given add it into its children.
        :param node_id: user id
        :param parent_id: parent user id
        :param act_time: node activation time
        :return: the created node
        """
        if node_id == parent_id:
            raise ValueError('node id and parent id must not be equal')
        if parent_id is None:
            node = CascadeNode(node_id, datetime=act_time)
            self.roots.append(node)
        else:
            parent = self.get_node(parent_id)
            if parent:
                node = CascadeNode(node_id, datetime=act_time, parent_id=parent_id)
                parent.children.append(node)
                if self.depth_of(parent_id) == self.depth:
                    self.depth += 1
            else:
                raise ValueError('parent node with user id given does not exist')
        self._id_to_node[node_id] = node
        return node


class ActSequence:
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
        u_time = self.times[index]
        while self.times[index + 1] == u_time:
            index += 1
        return self.users[:index]

    def get_rond_set(self, graph):
        # Return set of non-active nodes with at least one active parent node for each.
        if self.rond_set is None:
            result = set()
            for uid in set(self.users) & set(graph.nodes()):
                result.update(set(graph.successors(uid)))
            result = result - set(self.users)
            self.rond_set = result
        return self.rond_set

    def get_active_parents(self, uid, graph):
        rond_set = self.get_rond_set(graph)
        parents = set(graph.predecessors(uid)) if uid in graph else set()
        if uid in rond_set:
            active_parents = filter(lambda pid: pid in self.user_times, parents)
        else:
            u_time = self.user_times[uid]
            active_parents = filter(lambda pid: pid in self.user_times and self.user_times[pid] <= u_time, parents)
        # if str(uid) == '5d89465a86887712d4b704a9':
        #     logger.debug('parents = %s', parents)
        #     logger.debug('uid in rond_set = %s', uid in rond_set)
        #     logger.debug('active_parents = %s', list(active_parents))

        return list(active_parents)


class ParamTypes:
    JSON = 'json'
    ARRAY = 'array'
    SPARSE = 'sparse'
    SPARSE_LIST = 'splist'
    GRAPH = 'graph'


class GraphTypes:
    RESHARES = 0
    RELATIONS = 1


class Project:
    def __init__(self, project_name, db=None):
        self.name = project_name
        self.path = os.path.join(settings.BASE_PATH, 'data', project_name)
        if not os.path.exists(self.path):
            os.mkdir(self.path)  # Create the project path if it does not exist.
        try:
            self.db = self.load_param('db', ParamTypes.JSON)['db']
        except FileNotFoundError:
            if not db:
                raise ValueError('db is required if `db.json` does not exist')
            self.save_param({'db': db}, 'db', ParamTypes.JSON)
            self.db = db
        self.training = None
        self.test = None

    def save_sets(self, train_set, test_set):
        # Dump the json into the file.
        self.training = list(train_set)
        self.test = list(test_set)
        data = {
            'training': [str(cid) for cid in train_set],
            'test': [str(cid) for cid in test_set]
        }
        self.save_param(data, 'samples', ParamTypes.JSON)

    def load_sets(self):
        sample_set_path = os.path.join(self.path, 'samples.json')
        if os.path.exists(sample_set_path):
            data = json.load(open(sample_set_path))
            train_cascades, test_cascades = data['training'], data['test']
            self.training = [ObjectId(_id) for _id in train_cascades]
            self.test = [ObjectId(_id) for _id in test_cascades]
        else:
            raise Exception('Data sample not found. Run sampledata command.')

        return self.training, self.test

    @time_measure(level='debug')
    def load_trees(self):
        """
        Load trees of cascades in training and test sets.
        :return:
        """
        # Load trees from the json file.
        try:
            trees = self.load_param('trees', ParamTypes.JSON)
            trees = {ObjectId(key): value for key, value in trees.items()}
            # Convert tree dictionaries to tree objects.
            logger.debug('converting dictionaries to trees ...')
            trees = {cascades_id: CascadeTree().from_json(tree) for cascades_id, tree in trees.items()}
        except FileNotFoundError:
            logger.info('trees not found. extracting ...')
            trees = {}
            i = 0
            all_cascades = self.training + self.test
            count = len(all_cascades)
            for cascade_id in all_cascades:
                tree = self.extract_cascade(cascade_id)
                trees[cascade_id] = tree
                i += 1
                if i % 10 == 0:
                    logger.info('%d%% done', i * 100 / count)
            self.save_trees(trees)

        return trees

    def save_trees(self, trees):
        trees_dict = {str(cascade_id): tree.to_json() for cascade_id, tree in trees.items()}
        self.save_param(trees_dict, 'trees', ParamTypes.JSON)  # Save trees for the project.

    def extract_cascade(self, cascade_id):
        with Timer('TREE: fetching posts', level='debug'):
            # Fetch posts related to the cascade and reshares.
            if isinstance(cascade_id, str):
                cascade_id = ObjectId(cascade_id)
            db = DBManager(self.db).db
            post_ids = db.postcascades.distinct('post_id', {'cascade_id': cascade_id})
            posts = db.posts.find({'_id': {'$in': post_ids}}, {'url': 0}).sort('datetime')

            user_ids = list({p['author_id'] for p in posts})
            reshares = db.reshares.find({'post_id': {'$in': post_ids}, 'reshared_post_id': {'$in': post_ids}}) \
                .sort('datetime')

        with Timer('TREE: creating nodes', level='debug'):
            nodes = {user_id: CascadeNode(user_id) for user_id in user_ids}  # Create nodes for the users.
            visited = {uid: False for uid in user_ids}  # Set visited True if the node has been visited.

        with Timer('TREE: creating diffusion edges', level='debug'):
            """
            Create diffusion edge if a user reshares to another for the first time. Note that reshares are 
            sorted by time.
            """
            # logger.debug('TREE: reshares count = %d' % reshares.count())
            roots = []
            for reshare in reshares:
                child_id = reshare['user_id']
                parent_id = reshare['ref_user_id']
                if child_id == parent_id:
                    continue  # Continue if the 'reshare' is between same users.
                parent = nodes[parent_id]

                if not visited[parent_id]:  # It is a root
                    parent.datetime = reshare['ref_datetime']
                    visited[parent_id] = True
                    roots.append(parent)

                if not visited[child_id]:  # Any other node
                    child = nodes[child_id]
                    parent.children.append(child)
                    child.parent_id = parent_id
                    child.datetime = reshare['datetime']
                    visited[child_id] = True

        with Timer('TREE: Adding single nodes', level='debug'):
            # Add users with no diffusion edges as single nodes.
            posts.rewind()
            first_posts = {}
            for post in posts:
                if post['author_id'] not in first_posts:
                    first_posts[post['author_id']] = post

            for uid, node in nodes.items():
                if not visited[uid]:
                    post = first_posts[uid]
                    node.datetime = post['datetime']
                    roots.append(node)

        return CascadeTree(roots)

    @time_measure(level='debug')
    def load_or_extract_graph(self, train_set=None, post_ids=None):
        """
        Load or extract the graph of cascades in training set.
        If the graph is not saved in db and hence extracted and also if post_ids is a list, then post_ids is filled
        with the related post ids.
        :return:
        """
        if train_set is None:
            train_set, _ = self.load_sets()

        graph = None
        graph_info_fname = 'graph_info'
        try:
            graph_info = self.load_param(graph_info_fname, ParamTypes.JSON)
            train_set_strs = {str(cid) for cid in train_set}
            for fname, cascades in graph_info.items():
                if set(cascades) == train_set_strs:
                    try:
                        graph = self.load_param(fname, ParamTypes.GRAPH)
                        graph = relabel_nodes(graph, {n: ObjectId(n) for n in graph.nodes()})
                        break
                    except FileNotFoundError:  # If graph data does not exist.
                        pass
        except FileNotFoundError:
            graph_info = {}

        if graph is None:
            if post_ids is None:
                post_ids = []
            if len(post_ids) == 0:
                post_ids.extend(self._get_cascades_post_ids(train_set))
            graph = self.__extract_graph_from_reshares(post_ids, train_set)
            fname = 'graph' + str(max(int(name[5:]) for name in graph_info.keys()) + 1) if graph_info else 'graph1'
            logger.info('saving graph ...')
            self.save_param(graph, fname, ParamTypes.GRAPH)
            graph_info[fname] = [str(cid) for cid in train_set]
            self.save_param(graph_info, graph_info_fname, ParamTypes.JSON)

        return graph

    @time_measure(level='debug')
    def load_or_extract_act_seq(self, train_set=None, post_ids=None):
        """
        Load or extract list of activation sequences of cascades in training set.
        If the activation sequence is not saved in db and hence extracted and also if post_ids is a list, then post_ids
        is filled with the related post ids.
        :return:
        """
        seq_fname = 'sequences'
        all_train_set, _ = self.load_sets()
        if not train_set:
            train_set = all_train_set
        if post_ids is None:
            post_ids = []

        try:
            seq_copy = self.load_param(seq_fname, ParamTypes.JSON)
            sequences = {}
            for m in seq_copy:
                users = [ObjectId(item[0]) for item in seq_copy[m]['cascade']]
                times = [item[1] for item in seq_copy[m]['cascade']]
                sequences[ObjectId(m)] = ActSequence(users=users, times=times, max_t=seq_copy[m]['max_t'])

        except FileNotFoundError:  # If graph data does not exist.
            if len(post_ids) == 0:
                post_ids.extend(self._get_cascades_post_ids(all_train_set))
            sequences = self._extract_act_seq(post_ids, all_train_set)

        sequences = {cid: seq for cid, seq in sequences.items() if cid in train_set}

        return sequences

    @time_measure(level='debug')
    def load_or_extract_graph_seq(self, train_set=None):
        """
        Load the graph, and list of activation sequences of the project if saved before.
        Otherwise extract them from the training set and return them.
        :return: tuple of graph, and list of activation sequences
        """
        if not train_set:
            train_set, _ = self.load_sets()
        post_ids = []
        # if the graph is not saved in db and hence extracted, then post_ids is filled with the related post ids.
        graph = self.load_or_extract_graph(train_set, post_ids)
        sequences = self.load_or_extract_act_seq(train_set, post_ids)
        return graph, sequences

    @time_measure(level='debug')
    def _extract_act_seq(self, posts_ids, cascade_ids, seq_fname=None):
        """
        Extract list of activation sequences from given cascade id's.
        :param posts_ids:   list of posts id's
        :param cascade_ids:    cascade id's
        :param seq_fname:   the file name to save activation sequences data
        :return:            list of ActSequence's
        """
        logger.info('extracting act. sequences from %d posts ...' % len(posts_ids))
        post_count = len(posts_ids)
        users = {m: [] for m in cascade_ids}
        times = {m: [] for m in cascade_ids}

        db = DBManager(self.db).db

        # Iterate on posts to extract activation sequences.
        posts = self.__get_posts_author_datetime(posts_ids, db)
        i = 0
        for post in posts:
            if post['datetime'] is not None:
                for pc in db.postcascades.find({'post_id': post['_id'], 'cascade_id': {'$in': cascade_ids}},
                                               {'_id': 0, 'cascade_id': 1}):
                    cascade_id = pc['cascade_id']
                    if post['author_id'] not in users[cascade_id]:
                        users[cascade_id].append(post['author_id'])
                        times[cascade_id].append(post['datetime'])
            i += 1
            if i % (post_count // 10) == 0:
                logger.debug('%d%% posts done' % (i * 100 / post_count))

        logger.info('setting relative times and max times ...')
        max_t = {}
        i = 0
        for cascade in db.cascades.find({'_id': {'$in': cascade_ids}}, ['last_time', 'first_time']):
            mid = cascade['_id']
            times[mid] = [(t - cascade['first_time']).total_seconds() / (3600.0 * 24 * 30) for t in
                          times[mid]]  # number of months
            max_t[mid] = (cascade['last_time'] - cascade['first_time']).total_seconds() / (
                    3600.0 * 24 * 30)  # number of months
            i += 1
            if i % (len(cascade_ids) // 10) == 0:
                logger.debug('%d%% done' % (i * 100 / len(cascade_ids)))

        sequences = {}
        for m in cascade_ids:
            if users[m]:
                sequences[m] = ActSequence(users[m], times[m], max_t[m])

        logger.info('saving act. sequences ...')
        self.save_act_sequences(sequences, seq_fname)

        return sequences

    @time_measure(level='debug')
    def _get_cascades_post_ids(self, cascade_ids):
        logger.info('querying posts ids ...')
        db = DBManager(self.db).db
        post_ids = [pm['post_id'] for pm in
                    db.postcascades.find({'cascade_id': {'$in': cascade_ids}}, {'_id': 0, 'post_id': 1})]
        logger.debug('%d post ids fetched', len(post_ids))
        return post_ids

    def save_act_sequences(self, sequences, seq_fname=None):
        seq_copy = {}
        for m in sequences:
            seq_copy[str(m)] = {
                'cascade': [(str(sequences[m].users[i]), sequences[m].times[i]) for i in
                            range(len(sequences[m].users))],
                'max_t': sequences[m].max_t
            }
        if seq_fname is None:
            seq_fname = 'sequences'
        self.save_param(seq_copy, seq_fname, ParamTypes.JSON)

    def __get_posts_author_datetime(self, post_ids, db):
        max_count = 400000
        if len(post_ids) < max_count:
            logger.debug('fetching authors and datetimes of post ids ...')
            return list(db.posts.find({'_id': {'$in': post_ids}}, ['author_id', 'datetime'], no_cursor_timeout=True) \
                        .sort('datetime'))
        else:
            step = max_count
            posts = []
            for i in range(0, len(post_ids), step):
                logger.debug('fetching authors and datetimes of post ids %d - %d  ...', i,
                             min(i + step, len(post_ids)))
                posts.extend(list(db.posts.find({'_id': {'$in': post_ids[i: i + step]}}, ['author_id', 'datetime'],
                                                no_cursor_timeout=True)))
            posts.sort(key=lambda post: post['datetime'])
            return posts

    def __extract_reshares(self, db, post_ids):
        max_count = 400000
        if len(post_ids) < max_count:
            reshares = db.reshares.find(
                {'post_id': {'$in': post_ids}, 'reshared_post_id': {'$in': post_ids}},
                {'_id': 0, 'post_id': 1, 'reshared_post_id': 1, 'user_id': 1, 'ref_user_id': 1}).sort('datetime')
            return list(reshares)
        else:
            reshares = []
            step = max_count
            for i in range(0, len(post_ids), step):
                for j in range(0, len(post_ids), step):
                    logger.debug('fetching reshares from post ids %d - %d to post ids %d - %d  ...', i,
                                 min(i + step, len(post_ids)), j, min(j + step, len(post_ids)))
                    reshares.extend(list(db.reshares.find(
                        {'post_id': {'$in': post_ids[i: i + step]}, 'reshared_post_id': {'$in': post_ids[j:j + step]}},
                        {'_id': 0, 'post_id': 1, 'reshared_post_id': 1, 'user_id': 1, 'ref_user_id': 1,
                         'datetime': 1})))
                    logger.debug('number of reshares: %d', len(reshares))
            reshares.sort(key=lambda resh: resh['datetime'])
            return reshares

    @time_measure(level='debug')
    def __extract_graph_from_reshares(self, post_ids, cascade_ids):
        """
        Extract graph from given cascade id's.
        :param post_ids:    list of the post ids related to the cascades
        :param cascade_ids:    list of the cascade ids
        :return:            directed graph of all reshares
        """
        logger.info('extracting graph of reshares ...')

        logger.info('querying reshares ...')
        db = DBManager(self.db).db
        reshares = self.__extract_reshares(db, post_ids)
        resh_count = len(reshares)

        logger.info('extracting graph from %d posts and %d reshares ...', len(post_ids), resh_count)
        edges = []
        cascade_ids = set(cascade_ids)
        i = 0

        # Iterate on reshares to extract graph edges.
        for resh in reshares:
            user_id = resh['user_id']
            ref_user_id = resh['ref_user_id']
            if user_id != ref_user_id:
                src_cascade_ids = {pc['cascade_id'] for pc in
                                   db.postcascades.find({'post_id': resh['reshared_post_id']},
                                                        {'_id': 0, 'cascade_id': 1})}
                dest_cascade_ids = {pc['cascade_id'] for pc in
                                    db.postcascades.find({'post_id': resh['post_id']}, {'_id': 0, 'cascade_id': 1})}
                common_cascade = cascade_ids & src_cascade_ids & dest_cascade_ids
                if common_cascade:
                    edges.append((ref_user_id, user_id))
            i += 1
            if i % (resh_count / 10) == 0:
                logger.info('%d%% reshares done' % (i * 100 / resh_count))

        graph = DiGraph()
        graph.add_edges_from(edges)

        return graph

    def __extract_graph_from_trees(self, train_set):
        trees = self.load_trees()
        trees = [trees[cid] for cid in train_set] if train_set else list(trees.values())
        graph = DiGraph()
        i = 0
        for tree in trees:
            graph.add_edges_from(tree.edges())
            i += 1
            if i % 100 == 0:
                logger.debug('%d%% done', i / len(trees) * 100)
        return graph

    SUFFIXES = {
        ParamTypes.JSON: 'json',
        ParamTypes.ARRAY: 'npy',
        ParamTypes.SPARSE: 'npz',
        ParamTypes.SPARSE_LIST: 'npz',
        ParamTypes.GRAPH: 'txt'
    }

    def save_param(self, param, name, param_type):
        path = os.path.join(self.path, '%s.%s' % (name, self.SUFFIXES[param_type]))
        mkdir_rec(os.path.dirname(path))

        if param_type == ParamTypes.JSON:
            json.dump(param, open(path, 'w'), indent=1)
        elif param_type == ParamTypes.ARRAY:
            np.save(path, param)
        elif param_type == ParamTypes.SPARSE:
            save_sparse(path, param)
        elif param_type == ParamTypes.SPARSE_LIST:
            save_sparse_list(path, param)
        elif param_type == ParamTypes.GRAPH:
            write_adjlist(param, path)
        else:
            raise Exception('invalid type "%s"' % param_type)
        logger.debug('parameters saved in path %s', path)

    def load_param(self, name, type):
        path = os.path.join(self.path, '%s.%s' % (name, self.SUFFIXES[type]))
        logger.debug('loading parameters in path %s ...', path)
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

    def has_param(self, name, type):
        path = os.path.join(self.path, '%s.%s' % (name, self.SUFFIXES[type]))
        return os.path.exists(path)

    def delete_param(self, name, type):
        path = os.path.join(self.path, '%s.%s' % (name, self.SUFFIXES[type]))
        os.remove(path)
