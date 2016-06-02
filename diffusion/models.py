# -*- coding: utf-8 -*-
from datetime import timedelta
from difflib import get_close_matches
import logging
import re
import time
from crud.models import DiffusionParam, UserAccount, Post, Reshare, Meme, PostMeme
from utils.time_utils import str_to_datetime, DT_FORMAT

logger = logging.getLogger('social.diffusion.models')


class Generation(object):
    def __init__(self, posts):
        self.posts = posts
        self.summary = None

    def extract_summary(self):
        texts = [re.compile(r'\s*[.;!?؟]\s*', re.UNICODE).split(post.text) for post in self.posts]
        sentences = {}
        for post in texts:
            for sent in post:
                matches = get_close_matches(sent, sentences.keys())
                if matches:
                    sentences[matches[0]] += 1
                else:
                    sentences[sent] = 1
        thr = len(texts) / 4
        summary = u'. '.join([sent for sent in sentences if sentences[sent] > thr])
        return summary

    def get_summary(self):
        if not self.summary:
            self.summary = self.extract_summary()
        return self.summary

    def get_diff(self, prev_generation):
        summary = self.get_summary()
        sentences = re.compile(r'\s*[.;!?؟]\s*', re.UNICODE).split(summary)
        prev_summary = prev_generation.get_summary()
        prev_sentences = re.compile(r'\s*[.;!?؟]\s*', re.UNICODE).split(prev_summary)
        result = []
        matched = []
        for sent in sentences:
            matches = get_close_matches(sent, prev_sentences)
            if matches:
                color = 'black'
                matched.extend(matches)
            else:
                color = 'green'
            result.append({'text': sent, 'color': color})
        for sent in set(prev_sentences) - set(matched):
            result.append({'text': sent, 'color': 'red'})
        return result


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


class CascadePredictor(object):
    def __init__(self, tree):
        if not isinstance(tree, CascadeTree):
            raise ValueError('tree must be CascadeTree')
        self.tree = tree.copy()
        self.max_delay = 999999999

    def predict(self, log=False):
        if not self.tree:
            raise ValueError('no tree set')

        now = self.tree.max_datetime()  # Find the datetime of now.
        cur_step = sorted(self.tree.nodes(), key=lambda n: n['datetime'])  # Set tree nodes as initial step.
        cur_step_ids = [node['user']['id'] for node in cur_step]
        activated = self.tree.nodes()
        weight_sum = {}

        # Iterate on steps. For each step try to activate other nodes.
        i = 0
        while cur_step:
            i += 1
            if log:
                logger.info('\tstep %d ...' % i)
            edges = DiffusionParam.objects.filter(sender__in=cur_step_ids)
            out_edges = {edge.sender_id: [] for edge in edges}
            for edge in edges:
                out_edges[edge.sender_id].append(edge)
            next_step = []

            for node in cur_step:
                sender_id = node['user']['id']

                for edge in out_edges.get(sender_id, []):
                    if edge.receiver_id in activated:
                        continue
                    if edge.receiver_id not in weight_sum:
                        weight_sum[edge.receiver_id] = edge.weight
                    else:
                        weight_sum[edge.receiver_id] += edge.weight

                    #sample = random.random()
                    sample = 0.5
                    if sample <= weight_sum[edge.receiver_id]:
                        delay_param = edge.delay
                        if delay_param > self.max_delay:
                            continue
                        if delay_param < 0:  # Due to some rare bugs in delays
                            delay_param = -delay_param

                        #delay = np.random.exponential(delay_param)  # in days
                        delay = delay_param  # in days
                        send_dt = str_to_datetime(node['datetime'])
                        receive_dt = send_dt + timedelta(days=delay)
                        if receive_dt < now:
                            continue
                        child = CascadeTree.create_cascade_node(edge.receiver_id,
                                                                datetime=receive_dt.strftime(DT_FORMAT))
                        node['children'].append(child)
                        activated.append(edge.receiver_id)
                        next_step.append(child)
                        if log:
                            logger.info('\ta reshare predicted')
            cur_step = sorted(next_step, key=lambda n: n['datetime'])
            cur_step_ids = [node['user']['id'] for node in cur_step]

        return self.tree


class MemeDetector(object):
    def extract_memes(self, net=None):
        logger.info('collecting current memes ...')
        if net:
            memes = Meme.objects.filter(post__author__social_net=net)
        else:
            memes = Meme.objects.all()
        meme_texts = {meme.text: meme.id for meme in memes}

        post_memes = []
        count = 0

        if net:
            posts = Post.objects.filter(author__social_net=net)
        else:
            posts = Post.objects.all()
        total = posts.count()
        for post in posts.iterator():
            if post.text and not post.postmeme_set.exists():

                # Get the meme of the first ancestor if exists.
                parent = post.parents.all()[0].reshared_post if post.parents.exists() else None
                while parent and not parent.postmeme_set.exists():
                    parent = parent.parents.all()[0].reshared_post if parent.parents.exists() else None
                meme = None
                if parent:
                    meme = PostMeme.objects.filter(post=parent)[0].meme

                # If meme is not set, get it or save the new one.
                if not meme and post.text in meme_texts:
                    meme = Meme.objects.get(id=meme_texts[post.text])
                else:
                    meme = Meme.objects.create(text=post.text)
                    meme_texts[post.text] = meme.id
                    logger.info('new meme created')

                # Add to PostMeme's. Save them if multiplier of 1000.
                post_memes.append(PostMeme(post=post, meme=meme))
                if len(post_memes) % 1000 == 0:
                    logger.info('creating %d post memes ...' % len(post_memes))
                    PostMeme.objects.bulk_create(post_memes)
                    post_memes = []

            count += 1
            if count % 1000 == 0:
                logger.info('%d out of %d posts done' % (count, total))

        # Save remaining PostMeme's.
        if post_memes:
            logger.info('creating %d post memes ...' % len(post_memes))
            PostMeme.objects.bulk_create(post_memes)

        # Set the count and the first publication time of the memes.
        logger.info('setting count and first publication time for the memes ...')
        for meme in memes.iterator():
            meme_posts = posts.filter(postmeme__meme=meme).distinct()
            meme.count = meme_posts.count()
            if meme.count:
                meme.first_time = posts.order_by('datetime')[0].datetime
            meme.save()
