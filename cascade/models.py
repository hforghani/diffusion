# -*- coding: utf-8 -*-
from datetime import timedelta
import json
import logging
import os
import random
from networkx import DiGraph, read_adjlist, relabel_nodes, write_adjlist
from sklearn.preprocessing import normalize
from scipy import sparse
import time
from django.conf import settings
import numpy as np
from crud.models import UserAccount, Post, Reshare, Meme
from utils.numpy_utils import load_sparse, save_sparse, save_sparse_list, load_sparse_list
from utils.time_utils import str_to_datetime, DT_FORMAT

logger = logging.getLogger('diffusion.diffusion.models')


class CascadeTree(object):
    def __init__(self, tree=None):
        if tree:
            self.tree = {'children': tree}
        else:
            self.tree = {'children': []}

    def extract_cascade(self, meme, log=False):
        # Fetch posts related to the meme and reshares.
        t1 = time.time()
        posts = Post.objects.filter(postmeme__meme=meme).distinct().order_by('datetime')
        user_ids = posts.values_list('author__id', flat=True).distinct()
        reshares = Reshare.objects.filter(post__in=posts, reshared_post__in=posts).distinct().order_by('datetime')
        if log:
            logger.info('\tTREE: time 1 = %f' % (time.time() - t1))
            logger.info('\tTREE: reshares count = %d' % reshares.count())

        # Create nodes for the users.
        t1 = time.time()
        nodes = {}
        visited = {uid: False for uid in user_ids}  # Set visited True if the node has been visited.
        for uid in user_ids:
            nodes[uid] = CascadeTree.create_cascade_node(uid)
        if log:
            logger.info('\tTREE: time 2 = %f' % (time.time() - t1))

        # Create diffusion edge if a user reshares to another for the first time. Note that reshares are sorted by time.
        t1 = time.time()
        res = []
        for reshare in reshares:
            child_id = reshare.user_id
            parent_id = reshare.ref_user_id
            if child_id == parent_id:
                continue  # Continue if reshare is between the same user.
            parent = nodes[parent_id]

            if not visited[parent_id]:  # It is a root
                parent['post_id'] = reshare.reshared_post_id
                parent['datetime'] = reshare.ref_datetime.strftime(DT_FORMAT)
                visited[parent_id] = True
                res.append(parent)

            if not visited[child_id]:  # Any other node
                child = nodes[child_id]
                parent['children'].append(child)
                child['parent'] = parent_id
                child['post_id'] = reshare.post_id
                child['datetime'] = reshare.datetime.strftime(DT_FORMAT)
                visited[child_id] = True
        if log:
            logger.info('\tTREE: time 3 = %f' % (time.time() - t1))

        # Add users with no diffusion edges as single nodes.
        t1 = time.time()
        #datetimes = posts.values('author_id').annotate(min=Min('datetime'))
        #datetimes = {dt['author_id']: dt['min'] for dt in datetimes}
        first_posts = {}
        for post in posts:
            if post.author_id not in first_posts:
                first_posts[post.author_id] = post
        for uid, node in nodes.items():
            if not visited[uid]:
                #datetime = datetimes[uid]
                #node['datetime'] = datetime.strftime(DT_FORMAT)
                #first_post = posts.filter(author_id=node['user']['id'], datetime=datetime)[0]
                post = first_posts[uid]
                node['datetime'] = post.datetime.strftime(DT_FORMAT)
                node['post_id'] = post.id
                res.append(node)
        if log:
            logger.info('\tTREE: time 4 = %f' % (time.time() - t1))
        self.tree = {'children': res}
        return self

    @staticmethod
    def create_cascade_node(user_id, datetime=None, post_id=None, parent=None):
        return {'user': UserAccount.objects.get(id=user_id).get_dict(),
                'datetime': datetime,
                'post_id': post_id,
                'parent': parent,
                'children': []}

    def get_dict(self):
        return self.tree['children']

    def max_datetime(self, node=None):
        if node is None:
            node = self.tree
        max_dt = str_to_datetime(node['datetime']) if 'datetime' in node and node['datetime'] else None
        for child in node['children']:
            if max_dt is None:
                max_dt = self.max_datetime(child)
            else:
                max_dt = max(max_dt, self.max_datetime(child))
        return max_dt

    def get_leaves(self, node=None):
        if node is None:
            node = self.tree
        leaves = []
        if not node['children']:
            leaves.append(node)
        else:
            for child in node['children']:
                leaves.extend(self.get_leaves(child))
        return leaves

    def node_ids(self):
        return [node['user']['id'] for node in self.nodes()]

    def nodes(self, node=None):
        if node is None:
            node = self.tree
        if 'user' in node:
            nodes_list = [node]
        else:
            nodes_list = []
        for child in node['children']:
            nodes_list.extend(self.nodes(child))
        return nodes_list

    def edges(self, node=None):
        if node is None:
            node = self.tree
        if 'user' in node:
            parent_id = node['user']['id']
            edges_list = [(parent_id, child['user']['id']) for child in node['children']]
        else:
            edges_list = []
        for child in node['children']:
            edges_list.extend(self.edges(child))
        return edges_list

    def copy(self):
        copy_tree = self.copy_tree_dict()
        return CascadeTree(copy_tree['children'])

    def copy_tree_dict(self, node=None):
        if node is None:
            node = self.tree
        copy_node = {key: value for key, value in node.items()}
        if 'user' in copy_node:
            copy_node['user'] = {key: value for key, value in copy_node['user'].items()}
        copy_node['children'] = []
        for child in node['children']:
            copy_node['children'].append(self.copy_tree_dict(child))
        return copy_node


class AsLT(object):
    def __init__(self):
        self.max_delay = 999999999
        self.save_paths = {}  # implement in children.

    def fit(self, tree):
        if not isinstance(tree, CascadeTree):
            raise ValueError('tree must be CascadeTree')
        self.tree = tree.copy()
        return self

    def predict(self, log=False):
        if not self.tree:
            raise ValueError('no tree set')

        now = self.tree.max_datetime()  # Find the datetime of now.
        cur_step = sorted(self.tree.nodes(), key=lambda n: n['datetime'])  # Set tree nodes as initial step.
        activated = self.tree.nodes()
        weight_sum = {}
        w = load_sparse(self.save_paths['w'])
        r = np.load(self.save_paths['r'])
        user_ids = UserAccount.objects.values_list('id', flat=True).order_by('id')
        user_map = {user_ids[i]: i for i in range(len(user_ids))}

        # Iterate on steps. For each step try to activate other nodes.
        i = 0
        while cur_step:
            i += 1
            if log:
                logger.info('\tstep %d ...' % i)

            next_step = []

            for node in cur_step:
                u = node['user']['id']  # sender user id
                u_i = user_map[u]
                w_u = w[u_i, :]  # weights of the children of u

                # Iterate on children of u
                for v_i in np.nonzero(w_u)[1]:
                    v = user_ids[v_i]  # receiver (child) user id
                    if v in activated:
                        continue
                    if v not in weight_sum:
                        weight_sum[v] = w_u[v_i]
                    else:
                        weight_sum[v] += w_u[v_i]

                    sample = random.random()
                    #sample = 0.5
                    if sample <= weight_sum[v]:
                        delay_param = r[v_i]
                        if delay_param > self.max_delay:
                            continue
                        if delay_param < 0:  # Due to some rare bugs in delays
                            delay_param = -delay_param

                        delay = np.random.exponential(delay_param)  # in days
                        #delay = delay_param  # in days
                        send_dt = str_to_datetime(node['datetime'])
                        receive_dt = send_dt + timedelta(days=delay)
                        if receive_dt < now:
                            continue
                        child = CascadeTree.create_cascade_node(v, datetime=receive_dt.strftime(DT_FORMAT))
                        node['children'].append(child)
                        activated.append(v)
                        next_step.append(child)
                        if log:
                            logger.info('\ta reshare predicted')
            cur_step = sorted(next_step, key=lambda n: n['datetime'])

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


class Saito(AsLT):
    def __init__(self):
        super(Saito, self).__init__()
        self.save_paths = {var: os.path.join(settings.BASEPATH, 'resources', 'saito_%s.npz' % var) for var in
                           ['w', 'r', 'h', 'g', 'phi_h', 'phi_g', 'psi']}
        self.save_paths['r'] = self.save_paths['r'][:-3] + 'npy'
        self.sample_count = 1000

    def calc_parameters(self, iterations=3):
        # Load dataset.
        logger.info('extracting data ...')
        graph, data, meme_ids = self.load_or_extract_data()

        # Create db id to matrix id maps for users and memes.
        logger.info('creating user and meme id maps ...')
        user_ids = UserAccount.objects.values_list('id', flat=True).order_by('id')
        user_map = {user_ids[i]: i for i in range(len(user_ids))}
        meme_map = {meme_ids[i]: i for i in range(len(meme_ids))}

        # Set initial values of w and r.
        if os.path.exists(self.save_paths['w']) and os.path.exists(self.save_paths['r']):
            w = load_sparse(self.save_paths['w'])
            r = np.load(self.save_paths['r'])
            logger.info('w and r loaded')
        else:
            logger.info('initializing parameters ...')
            w, r = self.set_initial_values(graph, user_ids, user_map)
            save_sparse(self.save_paths['w'], w)
            np.save(self.save_paths['r'], r)

        # Run EM algorithm.
        logger.info('running algorithm ...')
        for i in range(iterations):
            t0 = time.time()

            logger.info('#%d' % (i + 1))
            if os.path.exists(self.save_paths['h']):
                h = load_sparse(self.save_paths['h'])
                logger.info('\th loaded')
            else:
                logger.info('\tcalculating h ...')
                h = self.calc_h(data, graph, w, r, meme_ids, meme_map, user_map)
                save_sparse(self.save_paths['h'], h)

            if os.path.exists(self.save_paths['g']):
                g = load_sparse(self.save_paths['g'])
                logger.info('\tg loaded')
            else:
                logger.info('\tcalculating g ...')
                g = self.calc_g(data, graph, w, r, meme_ids, meme_map, user_map)
                save_sparse(self.save_paths['g'], g)

            if not os.path.exists(self.save_paths['phi_h']):
                logger.info('\tcalculating phi_h ...')
                phi_h = self.calc_phi_h(data, graph, w, r, h, meme_ids, meme_map, user_map)
                save_sparse_list(self.save_paths['phi_h'], phi_h)
                del phi_h

            if not os.path.exists(self.save_paths['phi_g']):
                logger.info('\tcalculating phi_g ...')
                phi_g = self.calc_phi_g(data, graph, w, g, meme_ids, meme_map, user_map)
                save_sparse_list(self.save_paths['phi_g'], phi_g)
                del phi_g

            if not os.path.exists(self.save_paths['psi']):
                logger.info('\tcalculating psi ...')
                psi = self.calc_psi(data, graph, w, r, g, meme_ids, meme_map, user_map)
                save_sparse_list(self.save_paths['psi'], psi)
            else:
                psi = load_sparse_list(self.save_paths['psi'])
                logger.info('\tpsi loaded')

            del h
            del g
            phi_h = load_sparse_list(self.save_paths['phi_h'])
            logger.info('\tphi_h loaded')

            logger.info('\testimating r ...')
            last_r = r
            r = self.calc_r(data, graph, phi_h, psi, user_ids, meme_ids, meme_map, user_map)

            phi_g = load_sparse_list(self.save_paths['phi_g'])
            logger.info('\tphi_g loaded')

            logger.info('\testimating w ...')
            last_w = w
            w = self.calc_w(data, graph, phi_h, phi_g, psi, user_ids, user_map, meme_ids, meme_map)

            del phi_h
            del phi_g
            del psi
            np.save(self.save_paths['r'], r)
            save_sparse(self.save_paths['w'], w)

            for var in ['h', 'g', 'phi_h', 'phi_g', 'psi']:
                os.remove(self.save_paths[var])

            r_dif = np.linalg.norm(r - last_r)
            w_dif = w - last_w
            w_dif = np.sqrt(w_dif.multiply(w_dif).sum())
            logger.info('\tr dif = %s, w dif = %s' % (r_dif, w_dif))
            logger.info('\tr nnz = %d, w nnz = %d' % (np.count_nonzero(r), w.nnz))
            del last_r
            del last_w
            del r_dif
            del w_dif

            logger.info('\t\titeration time: %.2f min' % ((time.time() - t0) / 60.0))

    def load_or_extract_data(self):
        graph_path = os.path.join(settings.BASEPATH, 'resources', 'graph.txt')
        data_path = os.path.join(settings.BASEPATH, 'resources', 'cascades.json')
        train_set_path = os.path.join(settings.BASEPATH, 'resources', 'samples.json')

        if os.path.exists(train_set_path) and os.path.exists(graph_path) and os.path.exists(data_path):
            # Load graph and cascade data if exists.
            logger.info('\tloading data ...')
            train_set = json.load(open(train_set_path, 'r'))
            graph = DiGraph()
            graph = read_adjlist(graph_path, create_using=graph)
            graph = relabel_nodes(graph, {n: int(n) for n in graph.nodes()})
            data_copy = json.load(open(data_path, 'r'))
            data = {}
            for m in data_copy:
                users = [item[0] for item in data_copy[m]['cascade']]
                times = [item[1] for item in data_copy[m]['cascade']]
                data[int(m)] = CascadeData(users=users, times=times, max_t=data_copy[m]['max_t'])

        else:
            if self.sample_count:
                train_set = list(
                    np.random.choice(Meme.objects.filter(count__gt=500).values_list('id', flat=True), self.sample_count,
                                     replace=False))
            else:
                train_set = Meme.objects.filter(count__gt=500).values_list('id', flat=True)
                #train_set = json.load(open(train_set_path, 'r'))
            logger.info('\tquerying posts and reshares ...')
            t0 = time.time()
            posts = Post.objects.filter(postmeme__meme_id__in=train_set).distinct().order_by('datetime')
            reshares = Reshare.objects.filter(post__in=posts, reshared_post__in=posts).distinct().order_by('datetime')
            #posts = Post.objects.order_by('datetime')
            #reshares = Reshare.objects.order_by('datetime')
            resh_count = reshares.count()
            post_count = posts.count()
            logger.info('\t\ttime: %.2f min' % ((time.time() - t0) / 60.0))

            # Create dictionary of meme first times.
            logger.info('\textracting first times ...')
            first_times = Meme.objects.values('id', 'first_time')
            first_times = {obj['id']: obj['first_time'] for obj in first_times}

            # Create graph and cascade data.
            logger.info('\textracting cascades from %d posts and %d reshares ...' % (post_count, resh_count))
            meme_ids = Meme.objects.order_by('id').values_list('id', flat=True)
            edges, data = self.extract_data(posts, reshares, first_times, meme_ids)
            graph = DiGraph()
            graph.add_edges_from(edges)

            logger.info('\tsetting max times ...')
            i = 0
            for meme in Meme.objects.filter(id__in=data.keys()).iterator():
                data[meme.id].max_t = (meme.last_time - meme.first_time).total_seconds() / (
                    3600.0 * 24)  # number of days
                i += 1
                if i % (len(meme_ids) / 10) == 0:
                    logger.info('\t\t%d%% done' % (i * 100 / len(meme_ids)))

            logger.info('\tsaving data ...')
            data_copy = {}
            for m in data:
                data_copy[m] = {
                    'cascade': [(data[m].users[i], data[m].times[i]) for i in range(len(data[m].users))],
                    'max_t': data[m].max_t
                }
            json.dump(data_copy, open(data_path, 'w'), indent=4)
            json.dump(train_set, open(train_set_path, 'w'), indent=4)
            del data_copy
            write_adjlist(graph, graph_path)

        return graph, data, train_set

    def extract_data(self, posts, reshares, first_times, meme_ids):
        t0 = time.time()
        edges = []
        meme_ids = set(meme_ids)
        resh_count = reshares.count()
        i = 0

        # Iterate on reshares to extract graph edges.
        for resh in reshares.all():
            user_id = resh.user_id
            ref_user_id = resh.ref_user_id
            if user_id != ref_user_id:
                common_memes = meme_ids & set(resh.reshared_post.postmeme_set.values_list('meme_id', flat=True)) & set(
                    resh.post.postmeme_set.values_list('meme_id', flat=True))
                if common_memes:
                    edges.append((ref_user_id, user_id))
            i += 1
            if i % (resh_count / 10) == 0:
                logger.info('\t\t%d%% reshares done' % (i * 100 / resh_count))

        del reshares
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
                data[m] = CascadeData(users[m], times[m])
        logger.info('\t\ttime: %.2f min' % ((time.time() - t0) / 60.0))
        return edges, data

    def set_initial_values(self, graph, user_ids, user_map):
        t0 = time.time()
        u_count = len(user_ids)
        nodes = graph.nodes()
        values = []
        rows = []
        cols = []
        i = 0
        for v_i in range(u_count):
            v = user_ids[v_i]
            if v in nodes:
                parents = graph.predecessors(v)
                parents.append(v)
                par_indexes = [user_map[uid] for uid in parents]
                values.extend([1.0 / len(parents)] * len(parents))
                rows.extend(par_indexes)
                cols.extend([v_i] * len(parents))
            else:
                values.append(1)
                rows.append(v_i)
                cols.append(v_i)
            i += 1
            if i % (u_count / 10) == 0:
                logger.info('\t%d%% done' % (i * 100 / u_count))

        w = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float32)
        r = np.ones(u_count, np.float32)
        logger.info('\ttime: %.2f min' % ((time.time() - t0) / 60.0))
        return w, r

    def calc_h(self, data, graph, w, r, meme_ids, meme_map, user_map):
        t0 = time.time()
        u_count = len(user_map)
        m_count = len(meme_ids)
        values = []
        rows = []
        cols = []

        i = 0
        for mid in meme_ids:
            mid_i = meme_map[mid]
            cascade = data[mid]
            for uid in cascade.users:
                uid_i = user_map[uid]
                val = 0
                if cascade.user_times[uid] == cascade.times[0]:
                    val = 1
                else:
                    active_parents = cascade.get_active_parents(uid, graph)
                    if active_parents:
                        act_par_indexes = [user_map[id] for id in active_parents]
                        act_par_times = np.matrix([[cascade.user_times[pid] for pid in active_parents]])
                        user_time = np.repeat(np.matrix([cascade.user_times[uid]]), len(active_parents))
                        diff = user_time - act_par_times
                        diff[diff == 0] = 1.0 / (24 * 60)  # 1 minute
                        w_col = w[:, uid_i].todense()
                        val = float(np.exp(-r[uid_i] * diff) * w_col[act_par_indexes] * r[uid_i])
                        if np.float32(val) == 0:
                            logger.info('\t\tWARNING: h = 0')

                if val:
                    values.append(val)
                    rows.append(mid_i)
                    cols.append(uid_i)
            i += 1
            if i % (m_count / 10) == 0:
                logger.info('\t\t%d%% done' % (i * 100 / m_count))

        h = sparse.csc_matrix((values, [rows, cols]), shape=(m_count, u_count), dtype=np.float32)
        logger.info('\t\ttime: %.2f min' % ((time.time() - t0) / 60.0))
        return h

    def calc_g(self, data, graph, w, r, meme_ids, meme_map, user_map):
        t0 = time.time()
        u_count = len(user_map)
        m_count = len(meme_ids)
        values = []
        rows = []
        cols = []
        i = 0

        for mid in meme_ids:
            mid_i = meme_map[mid]
            cascade = data[mid]
            rond_set = cascade.get_rond_set(graph)

            for uid in rond_set:
                uid_i = user_map[uid]
                active_parents = cascade.get_active_parents(uid, graph)
                act_par_indexes = [user_map[id] for id in active_parents]
                inactive_parents = set(graph.predecessors(uid)) - set(active_parents)
                inact_par_indexes = [user_map[id] for id in inactive_parents]

                w_col = w[:, uid_i].todense()
                inact_par_sum = w_col[inact_par_indexes].sum()
                act_par_times = np.matrix([[cascade.user_times[pid] for pid in active_parents]])
                max_time = np.repeat(np.matrix(cascade.max_t), len(active_parents))
                diff = max_time - act_par_times
                act_par_sum = np.exp(-r[uid_i] * diff) * w_col[act_par_indexes]

                val = w[uid_i, uid_i] + inact_par_sum + float(act_par_sum)
                if np.float32(val) == 0:
                    logger.info('\t\tWARNING: g = 0')
                values.append(val)
                rows.append(mid_i)
                cols.append(uid_i)

            i += 1
            if i % (len(meme_ids) / 10) == 0:
                logger.info('\t\t%d%% done' % (i * 100 / len(meme_ids)))

        g = sparse.csc_matrix((values, [rows, cols]), shape=(m_count, u_count), dtype=np.float32)
        logger.info('\t\ttime: %.2f min' % ((time.time() - t0) / 60.0))
        return g

    def calc_phi_h(self, data, graph, w, r, h, meme_ids, meme_map, user_map):
        t0 = time.time()
        u_count = len(user_map)
        phi_h = [None for _ in range(len(meme_ids))]
        i = 0

        for mid in meme_ids:
            mid_i = meme_map[mid]
            cascade = data[mid]
            values = []
            rows = []
            cols = []

            for v in cascade.users:
                v_i = user_map[v]
                active_parents = cascade.get_active_parents(v, graph)
                if not active_parents:
                    continue
                act_par_indexes = [user_map[id] for id in active_parents]
                act_par_times = np.matrix([[cascade.user_times[pid] for pid in active_parents]])
                user_time = np.repeat(np.matrix([cascade.user_times[v]]), len(active_parents))
                diff = user_time - act_par_times
                diff[diff == 0] = 1.0 / (24 * 60)  # 1 minute
                w_col = w[:, v_i].todense()
                val = np.multiply(w_col[act_par_indexes].T, np.exp(-r[v_i] * diff)) * r[v_i] / h[mid_i, v_i]
                if np.isinf(np.float32(val)).any():
                    logger.info('\t\tWARNING: phi_h = inf')
                    #if (np.float32(val) == 0).any():
                #    logger.info('\t\tWARNING: phi_h = 0')
                if val.size > 1:
                    values.extend(list(np.array(val).squeeze()))
                else:
                    values.append(float(val))
                rows.extend(act_par_indexes)
                cols.extend([v_i] * len(act_par_indexes))

            phi_h[mid_i] = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float32)

            i += 1
            if i % (len(meme_ids) / 10) == 0:
                logger.info('\t\t%d%% done' % (i * 100 / len(meme_ids)))

        logger.info('\t\ttime: %.2f min' % ((time.time() - t0) / 60.0))
        return phi_h

    def calc_phi_g(self, data, graph, w, g, meme_ids, meme_map, user_map):
        t0 = time.time()
        u_count = len(user_map)
        phi_g = [None for _ in range(len(meme_ids))]
        i = 0

        for mid in meme_ids:
            mid_i = meme_map[mid]
            cascade = data[mid]
            values = []
            rows = []
            cols = []

            for v in cascade.get_rond_set(graph):
                v_i = user_map[v]
                u_set = {v} | (set(graph.predecessors(v)) - set(cascade.get_active_parents(v, graph)))
                if not u_set:
                    continue
                u_indexes = [user_map[id] for id in u_set]
                w_col = w[:, v_i].todense()
                val = w_col[u_indexes] / g[mid_i, v_i]
                if np.isinf(np.float32(val)).any():
                    logger.info('\t\tWARNING: phi_g = inf')
                    #if (np.float32(val) == 0).any():
                #    logger.info('\t\tWARNING: phi_g = 0')
                if val.size > 1:
                    values.extend(list(np.array(val).squeeze()))
                else:
                    values.append(float(val))
                rows.extend(u_indexes)
                cols.extend([v_i] * len(u_indexes))

            phi_g[mid_i] = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float32)

            i += 1
            if i % (len(meme_ids) / 10) == 0:
                logger.info('\t\t%d%% done' % (i * 100 / len(meme_ids)))

        logger.info('\t\ttime: %.2f min' % ((time.time() - t0) / 60.0))
        return phi_g

    def calc_psi(self, data, graph, w, r, g, meme_ids, meme_map, user_map):
        t0 = time.time()
        u_count = len(user_map)
        psi = [None for _ in range(len(meme_ids))]
        i = 0

        for mid in meme_ids:
            mid_i = meme_map[mid]
            cascade = data[mid]
            values = []
            rows = []
            cols = []

            for v in cascade.get_rond_set(graph):
                v_i = user_map[v]
                active_parents = cascade.get_active_parents(v, graph)
                act_par_indexes = [user_map[id] for id in active_parents]
                act_par_times = np.matrix([[cascade.user_times[pid] for pid in active_parents]])
                max_time = np.repeat(np.matrix([cascade.max_t]), len(active_parents))
                diff = max_time - act_par_times
                w_col = w[:, v_i].todense()
                val = np.multiply(w_col[act_par_indexes].T, np.exp(-r[v_i] * diff)) / g[mid_i, v_i]
                if np.isinf(np.float32(val)).any():
                    logger.info('\t\tWARNING: psi = inf')
                    #if (np.float32(val) == 0).any():
                #    logger.info('\t\tWARNING: psi = 0')
                if val.size > 1:
                    values.extend(list(np.array(val).squeeze()))
                else:
                    values.append(float(val))
                rows.extend(act_par_indexes)
                cols.extend([v_i] * len(act_par_indexes))

            psi[mid_i] = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype=np.float32)

            i += 1
            if i % (len(meme_ids) / 10) == 0:
                logger.info('\t\t%d%% done' % (i * 100 / len(meme_ids)))

        logger.info('\t\ttime: %.2f min' % ((time.time() - t0) / 60.0))
        return psi

    def calc_r(self, data, graph, phi_h, psi, user_ids, meme_ids, meme_map, user_map):
        t0 = time.time()
        u_count = len(user_ids)
        r = np.ones(u_count, np.float32)

        logger.info('\t\textracting sigma domains ...')
        m_set1 = {v: [] for v in user_ids}
        m_set2 = {v: [] for v in user_ids}
        for m in meme_ids:
            for v in data[m].users:
                m_set1[v].append(m)
            for v in data[m].get_rond_set(graph):
                m_set2[v].append(m)

        logger.info('\t\tcalculating values ...')
        i = 0
        for v in user_ids:
            v_i = user_map[v]

            phi_sum = 0
            phi_time_sum = 0
            psi_time_sum = 0
            for m in set(m_set1[v]) | set(m_set2[v]):
                m_i = meme_map[m]
                cascade = data[m]
                active_parents = cascade.get_active_parents(v, graph)
                if not active_parents:
                    continue
                act_par_indexes = [user_map[id] for id in active_parents]
                act_par_times = np.matrix([[cascade.user_times[pid] for pid in active_parents]])

                if m in m_set1[v]:
                    phi_h_col = phi_h[m_i][:, v_i].todense()
                    phi_sum += phi_h_col[act_par_indexes].sum()
                    user_time = np.repeat(np.matrix([cascade.user_times[v]]), len(active_parents))
                    diff = user_time - act_par_times
                    diff[diff == 0] = 1.0 / (24 * 60)  # 1 minute
                    phi_time_sum += float(diff * phi_h_col[act_par_indexes])

                if m in m_set2[v]:
                    psi_col = psi[m_i][:, v_i]
                    max_time = np.repeat(np.matrix([cascade.max_t]), len(active_parents))
                    diff = max_time - act_par_times
                    psi_time_sum += float(diff * psi_col[act_par_indexes])

            if phi_sum == 0:
                r[v_i] = 0
                #if m_set1[v] or m_set2[v]:
                #    logger.info('\t\tWARNING: r = 0, sets: %s, %s' % (m_set1[v], m_set2[v]))
            else:
                if phi_time_sum + psi_time_sum != 0:
                    r[v_i] = phi_sum / (phi_time_sum + psi_time_sum)
                else:
                    r[v_i] = np.finfo(np.float32).max
                    logger.info('\t\tWARNING: denominator = 0, r = inf')

            i += 1
            if i % (len(user_ids) / 10) == 0:
                logger.info('\t\t%d%% done' % (i * 100 / len(user_ids)))

        logger.info('\t\ttime: %.2f min' % ((time.time() - t0) / 60.0))
        return r

    def calc_w(self, data, graph, phi_h, phi_g, psi, user_ids, user_map, meme_ids, meme_map):
        t0 = time.time()
        u_count = len(user_ids)

        logger.info('\t\textracting sigma domains ...')
        mv_set2 = {v: [] for v in graph.nodes()}
        muv_set1 = {edge: [] for edge in graph.edges()}
        muv_set2 = {edge: [] for edge in graph.edges()}
        muv_set3 = {edge: [] for edge in graph.edges()}
        for m in meme_ids:
            cascade = data[m]
            for v in cascade.users:
                for u in cascade.get_active_parents(v, graph):
                    muv_set1[(u, v)].append(m)
            for v in cascade.get_rond_set(graph):
                mv_set2[v].append(m)
                for u in set(graph.predecessors(v)) - set(cascade.get_active_parents(v, graph)):
                    muv_set2[(u, v)].append(m)
                for u in cascade.get_active_parents(v, graph):
                    muv_set3[(u, v)].append(m)

        logger.info('\t\tcalculating values ...')
        values = []
        rows = []
        cols = []
        val_count = len(graph.edges()) + len(graph.nodes())
        i = 0
        for (u, v) in graph.edges():
            u_i = user_map[u]
            v_i = user_map[v]
            phi_h_sum = 0
            phi_g_sum = 0
            psi_sum = 0
            if muv_set1[(u, v)]:
                phi_h_sum = np.array([phi_h[meme_map[m]][u_i, v_i] for m in muv_set1[(u, v)]]).sum()
            if muv_set2[(u, v)]:
                phi_g_sum = np.array([phi_g[meme_map[m]][u_i, v_i] for m in muv_set2[(u, v)]]).sum()
            if muv_set3[(u, v)]:
                psi_sum = np.array([psi[meme_map[m]][u_i, v_i] for m in muv_set3[(u, v)]]).sum()
            val = phi_h_sum + phi_g_sum + psi_sum
            if val:
                values.append(val)
                rows.append(u_i)
                cols.append(v_i)
                #elif muv_set1[(u, v)] or muv_set2[(u, v)] or muv_set3[(u, v)]:
            #    logger.info('\t\tWARNING: w = 0 at %s, sets: %s, %s, %s' % (
            #        (u, v), muv_set1[(u, v)], muv_set2[(u, v)], muv_set3[(u, v)]))

            i += 1
            if i % (val_count / 10) == 0:
                logger.info('\t\t%d%% done' % (i * 100 / val_count))

        for v in graph.nodes():
            v_i = user_map[v]
            if mv_set2[v]:
                phi_g_sum = np.array([phi_g[meme_map[m]][v_i, v_i] for m in mv_set2[v]]).sum()
                if phi_g_sum:
                    values.append(phi_g_sum)
                    rows.append(v_i)
                    cols.append(v_i)
                    #else:
                    #    logger.info('\t\tWARNING: w = 0 at %s, set: %s' % ((v, v), mv_set2[v]))

            i += 1
            if i % (val_count / 10) == 0:
                logger.info('\t\t%d%% done' % (i * 100 / val_count))

        w = sparse.csc_matrix((values, [rows, cols]), shape=(u_count, u_count), dtype='d')

        logger.info('\t\tnormalizing w ...')
        w = normalize(w, axis=0, copy=False)
        w = sparse.csc_matrix((w.data, w.indices, w.indptr), shape=w.shape, dtype=np.float32)

        logger.info('\t\ttime: %.2f min' % ((time.time() - t0) / 60.0))
        return w
