# -*- coding: utf-8 -*-
import os
import re
import traceback

from bulk_update.helper import bulk_update
from django.core.exceptions import ObjectDoesNotExist
from django.core.management.base import BaseCommand, CommandError
from django.db.models.aggregates import Count, Min, Max
import time
import pygtrie
from crud.models import UserAccount, SocialNet, Post, Reshare, Meme, PostMeme
from utils.time_utils import str_to_datetime


class Command(BaseCommand):
    help = 'Create database instances using memetracker dataset.'

    def add_arguments(self, parser):
        parser.add_argument(
            "file",
            type=str,
            nargs='?',
            help="Memetracker database text file path"
        )
        parser.add_argument(
            "-s",
            "--start",
            type=int,
            dest="start_index",
            help="determine which index of post in the dataset file to start from"
        )
        parser.add_argument(
            "-a",
            "--attributes",
            action="store_true",
            dest="set_attributes",
            help="just set attributes and ignore creating data"
        )
        parser.add_argument(
            "-c",
            "--clear",
            action="store_true",
            dest="clear",
            help="clear existing data and continue"
        )
        parser.add_argument(
            "-e",
            "--entities",
            action="store_true",
            dest="entities",
            help="just create meme, user_account, and post entities"
        )
        parser.add_argument(
            "-r",
            "--relations",
            action="store_true",
            dest="relations",
            help="just create post_meme and reshare relations"
        )
        parser.add_argument(
            "-t",
            "--text",
            action="store_true",
            dest="post_texts",
            help="Set post texts according to their memes"
        )

    def __init__(self):
        super(Command, self).__init__()
        self.memes_map = pygtrie.StringTrie()  # A trie data structure that maps from meme texts to ids

    def handle(self, *args, **options):
        try:
            start = time.time()

            path = options['file']
            if not path:
                raise CommandError('file argument not specified')

            # Concatenate memes of each post and set it as the post text.
            if options['post_texts']:
                self.stdout.write('setting post texts ...')
                posts = Post.objects.filter(text='', postmeme__isnull=False)
                count = posts.count()
                post_memes = posts.values('id', 'postmeme__meme__text').order_by('id')
                if count:
                    cur_post_id = post_memes[0]['id']
                    i = 0
                    memes = set()
                    for post_meme in post_memes.iterator():
                        if post_meme['id'] != cur_post_id:
                            Post.objects.filter(id=cur_post_id).update(text='. '.join(memes))
                            cur_post_id = post_meme['id']
                            memes = set()
                            i += 1
                            if i % 100 == 0:
                                self.stdout.write('%d / %d' % (i, count))
                        memes.add(post_meme['postmeme__meme__text'])
                    Post.objects.filter(id=cur_post_id).update(text='. '.join(memes))
                return

            if not options['set_attributes']:
                # Delete all data.
                if options['clear']:
                    self.stdout.write('======== deleting data ...')
                    PostMeme.objects.all().delete()
                    Reshare.objects.all().delete()
                    Post.objects.all().delete()
                    Meme.objects.all().delete()
                    UserAccount.objects.all().delete()
                    SocialNet.objects.all().delete()

                # Get or create social net.
                try:
                    net = SocialNet.objects.get(name='memetracker')
                except ObjectDoesNotExist:
                    net = SocialNet.objects.create(name='memetracker', icon='/media/img/memetracker.gif')

                # Create instances of non-relation entities.
                if options['entities'] or not options['relations']:
                    self.stdout.write('======== creating entities ...')
                    self.create_entities(path, net)

                # Create instances of relation entities.
                if options['relations'] or not options['entities']:
                    temp_data_path = os.path.join(os.path.dirname(path), os.path.basename(path) + '.temp')
                    if not os.path.exists(temp_data_path):
                        self.create_temp(path, temp_data_path)

                    self.stdout.write('======== creating relations ...')
                    self.create_relations(temp_data_path, start_index=options['start_index'])

                    # Set the meme count, first time, and last time attributes of memes.
                    self.stdout.write('======== setting counts and publication times for the memes ...')
                    self.calc_memes_values()

            self.stdout.write('======== command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise

    def create_entities(self, path, net):
        urls = {}
        memes = set()
        i = 0

        self.stdout.write('reading urls and memes from dataset ...')
        with open(path, encoding="utf8") as f:
            line = f.readline()

            while line:
                char = line[0]
                text = line[2:-1]
                if char == 'P':
                    post_url = self.truncate_url(text)
                    i += 1
                    if i % 1000000 == 0:
                        self.stdout.write('%d posts read' % i)
                elif char == 'T':  # time line
                    urls[post_url] = str_to_datetime(text)
                elif char == 'Q':
                    memes.add(text)
                elif char == 'L':
                    link_url = self.truncate_url(text)
                    if link_url not in urls:
                        urls[link_url] = None
                line = f.readline()
        self.stdout.write('%d posts read' % i)
        self.stdout.write('{} memes extracted from dataset'.format(len(memes)))

        self.stdout.write('loading existing memes ...')
        existing_memes = set(Meme.objects.values_list('text', flat=True))
        memes -= existing_memes
        del existing_memes
        self.stdout.write('creating %d new memes ...' % len(memes))
        meme_entities = []
        i = 0
        for text in memes:
            meme_entities.append(Meme(text=text))
            i += 1
            if i % 100000 == 0:
                Meme.objects.bulk_create(meme_entities)
                self.stdout.write('%d memes created' % i)
                meme_entities = []
        Meme.objects.bulk_create(meme_entities)
        self.stdout.write('%d memes created' % len(meme_entities))
        del memes
        del meme_entities

        self.stdout.write('loading existing urls ...')
        existing_urls = set(Post.objects.values_list('url', flat=True))
        urls = {key: value for key, value in urls.items() if key not in existing_urls}
        del existing_urls
        self.stdout.write('loading existing usernames ...')
        existing_usernames = set(UserAccount.objects.values_list('username', flat=True))
        usernames = set([self.get_username(url) for url in urls]) - existing_usernames - {None}
        del existing_usernames

        self.stdout.write('creating %d new users ...' % len(usernames))
        i = 0
        users = []
        for uname in usernames:
            users.append(UserAccount(username=uname, social_net=net))
            i += 1
            if i % 100000 == 0:
                UserAccount.objects.bulk_create(users)
                self.stdout.write('%d users created' % i)
                users = []
        UserAccount.objects.bulk_create(users)
        self.stdout.write('%d users created' % len(users))
        del usernames
        del users
        self.stdout.write('loading user ids ...')
        users = UserAccount.objects.values('username', 'id')
        users_map = {user['username']: user['id'] for user in users}
        del users

        # Create all posts with empty texts. Texts are set later in create_relations.
        self.stdout.write('creating %d new posts ...' % len(urls))
        posts = []
        i = 0
        t0 = time.time()
        for url, dt in urls.items():
            username = self.get_username(url)
            if username:
                user_id = users_map[username]
                posts.append(Post(url=url, text='', author_id=user_id, datetime=dt))
            i += 1
            if i % 10000 == 0:
                Post.objects.bulk_create(posts)
                posts = []
            if i % 10000 == 0:
                self.stdout.write('%d posts created (%.1f s)' % (i, (time.time() - t0)))
                t0 = time.time()
        Post.objects.bulk_create(posts)
        self.stdout.write('%d posts created (%.1f s)' % (i, (time.time() - t0)))
        del posts, urls, users_map

    # @profile
    def create_relations(self, temp_data_path, start_index):
        source_ids = []
        post_id = None
        datetime = None
        meme_ids = []
        post_memes = []
        reshares = []
        i = 0
        t0 = time.time()

        # if 'start_index' is specified, ignore lower indexes.
        ignoring = False
        if start_index:
            ignoring = True

        with open(temp_data_path, encoding="utf8") as f:
            line = f.readline()

            while line:
                char = line[0]
                text = line[2:-1]

                # Count posts.
                if char == 'P':
                    i += 1
                    if (not ignoring or i == start_index) and i % 1000 == 0:
                        self.stdout.write('processing posts: %d' % i)
                    elif ignoring and i % 100000 == 0:
                        self.stdout.write('ignoring posts: %d' % i)

                # Handle if it is in ignoring state.
                if ignoring:
                    if i <= start_index:
                        line = f.readline()
                        continue
                    else:
                        ignoring = False
                        t0 = time.time()

                if char == 'P':  # post line
                    if post_id is not None:
                        pm, resh = self.get_post_rels(post_id, datetime, meme_ids, source_ids)
                        post_memes.extend(pm)
                        reshares.extend(resh)
                        if i % 10000 == 0:
                            self.stdout.write(
                                'saving %d post memes and %d reshares ...' % (len(post_memes), len(reshares)))
                            PostMeme.objects.bulk_create(post_memes)
                            Reshare.objects.bulk_create(reshares)
                            post_memes = []
                            reshares = []
                            self.stdout.write('time : {:.2f} m'.format((time.time() - t0) / 60))
                            t0 = time.time()

                    try:
                        post_id = int(text)
                    except ValueError:
                        raise CommandError("invalid post id: '{}'".format(text))
                    source_ids = []
                    meme_ids = []
                elif char == 'T':  # time line
                    datetime = str_to_datetime(text)
                elif char == 'Q':  # meme line
                    try:
                        meme_id = int(text)
                    except ValueError:
                        self.stdout.write("meme '{}' ignored".format(text))
                    meme_ids.append(meme_id)
                elif char == 'L':  # link line
                    try:
                        source_ids.append(int(text))
                    except ValueError:
                        self.stdout.write("link '{}' ignored".format(text))

                line = f.readline()

        # Add the relations of the last post.
        pm, resh = self.get_post_rels(post_id, datetime, meme_ids, source_ids)
        post_memes.extend(pm)
        reshares.extend(resh)

        # Save the remaining relations.
        self.stdout.write(
            'saving %d post memes and %d reshares ...' % (len(post_memes), len(reshares)))
        PostMeme.objects.bulk_create(post_memes)
        Reshare.objects.bulk_create(reshares)

    def get_post_rels(self, post_id, datetime, meme_ids, source_ids):
        """
        Create PostMeme and Reshare instances for the referenced links. Just create the instances not inserting in the db.
        """
        # trunc_url = self.truncate_url(post_id)

        # Create the post.
        try:
            post = Post.objects.get(id=post_id)
        except ObjectDoesNotExist:
            raise CommandError('post does not exist with id {}'.format(post_id))

        # Assign the memes to the post.
        post_memes = [PostMeme(post_id=post_id, meme_id=mid) for mid in meme_ids]

        # Create reshares if the post is reshared.
        reshares = []
        src_ids = set(source_ids) - {post_id}
        src_posts = Post.objects.filter(id__in=src_ids)
        if len(src_posts) != len(src_ids):  # Raise an error if some of link posts do not exist.
            not_existing = src_ids - {p.id for p in src_posts}
            raise CommandError('link post does not exist with id(s): {}'.format(', '.join(not_existing)))
        for src_post in src_posts:
            reshares.append(
                    Reshare(post_id=post.id, reshared_post_id=src_post.id, datetime=datetime))

        return post_memes, reshares

    def create_temp(self, path, temp_path):
        # Replace meme texts with meme ids and create temporary data files.
        from_path = path
        memes_count = Meme.objects.count()
        step = 5 * 10 ** 6
        i = 0
        for offset in range(0, memes_count, step):
            to_path = '{}.memes{}'.format(temp_path, i)
            if not os.path.exists(to_path):
                end = min(offset + step, memes_count)
                self.stdout.write('loading memes map from {} to {} ...'.format(offset, end))
                memes_map = self.load_memes(offset, step)
                self.stdout.write('replacing meme texts with meme ids from {} to {} ...'.format(offset, end))
                self.replace(from_path, to_path, 'Q', memes_map)
                del memes_map
            i += 1
            from_path = to_path

        # Replace post urls with post ids and create temporary data files.
        posts_count = Post.objects.count()
        step = 5 * 10 ** 6
        i = 0
        for offset in range(0, posts_count, step):
            to_path = '{}.posts{}'.format(temp_path, i)
            if not os.path.exists(to_path):
                end = min(offset + step, posts_count)
                self.stdout.write('loading posts map from {} to {} ...'.format(offset, end))
                posts_map = self.load_posts(offset, step)
                self.stdout.write('replacing post urls with post ids from {} to {} ...'.format(offset, end))
                self.replace(from_path, to_path, 'PL', posts_map)
                del posts_map
            i += 1
            from_path = to_path

        os.rename(from_path, temp_path)

    def replace(self, in_path, out_path, characters, replace_map):
        in_batch_size = 100
        out_batch_size = 100

        with open(in_path, encoding="utf8") as fin:
            with open(out_path, 'w', encoding="utf8") as fout:
                in_lines = []
                out_lines = []

                while True:
                    if not in_lines:
                        in_lines = fin.readlines(in_batch_size)
                        if not in_lines:
                            break
                    line = in_lines.pop(0)
                    ch = line[0]
                    if ch in characters:
                        text = line[2:-1]
                        if ch in 'PL':
                            text = self.truncate_url(text)
                        if not re.match(r'\d+$', text) and text in replace_map:
                            out = '{}\t{}\n'.format(ch, replace_map[text])
                        else:
                            out = line
                    else:
                        out = line
                    out_lines.append(out)

                    if len(out_lines) >= out_batch_size:
                        fout.writelines(out_lines)
                        out_lines = []

                fout.writelines(out_lines)

    def load_memes(self, offset=0, limit=None):
        """
        Get map of meme texts to meme id's.
        :return:
        """
        memes_map = pygtrie.StringTrie()  # A trie data structure that maps from meme texts to ids
        memes = Meme.objects.values('text', 'id')
        if limit is not None or offset > 0:
            if limit is not None:
                memes = memes.order_by('id')[offset: offset + limit]
            else:
                memes = memes.order_by('id')[offset:]
        for meme in memes:
            memes_map[meme['text']] = meme['id']
        return memes_map

    def load_posts(self, offset=0, limit=None):
        """
        Get map of post urls to post id's
        :return:
        """
        posts_map = pygtrie.StringTrie()  # A trie data structure that maps from meme texts to ids
        posts = Post.objects.values('url', 'id')
        if limit is not None or offset > 0:
            if limit is not None:
                posts = posts.order_by('id')[offset: offset + limit]
            else:
                posts = posts.order_by('id')[offset:]
        for post in posts:
            posts_map[post['url']] = post['id']
        return posts_map

    def calc_memes_values(self):
        self.stdout.write('creating queries ...')
        meme_counts = PostMeme.objects.values('meme_id').annotate(count=Count('post'))
        first_times = PostMeme.objects.values('meme_id').annotate(first=Min('post__datetime'))
        last_times = PostMeme.objects.values('meme_id').annotate(last=Max('post__datetime'))
        self.stdout.write('calculating meme counts ...')
        meme_counts = {obj['meme_id']: obj['count'] for obj in meme_counts}
        self.stdout.write('calculating first pub. times of memes ...')
        first_times = {obj['meme_id']: obj['first'] for obj in first_times}
        self.stdout.write('calculating last pub. times of memes ...')
        last_times = {obj['meme_id']: obj['last'] for obj in last_times}

        self.stdout.write('saving ...')
        i = 0
        t0 = time.time()
        memes = []
        for meme in Meme.objects.iterator():
            mid = meme.id
            meme.count = meme_counts[mid] if mid in meme_counts else None
            meme.first_time = first_times[mid] if mid in first_times else None
            meme.last_time = last_times[mid] if mid in last_times else None
            memes.append(meme)
            i += 1
            if i % 1000 == 0:
                bulk_update(memes)
                memes = []
            if i % 100000 == 0:
                self.stdout.write('%d memes saved in %d s' % (i, time.time() - t0))
                t0 = time.time()
        else:
            bulk_update(memes)
            self.stdout.write('%d memes saved in %d s' % (i, time.time() - t0))

    def get_username(self, url):
        """
        Extract the username from the url. Consider the domain name as the username.
        :param url: url
        :return:    domain name as the username. Return None if the url is invalid.
        """
        try:
            return re.match(r'https?://+([^/?]*\w+[^/?]*)', url.lower()).groups()[0][:100]
        except AttributeError:
            return None

    def truncate_url(self, url):
        """
        Truncate the url to maximum 100 characters to save in DB.
        if the length is greater than 100, concatenate the first 50 and the last 50 characters.
        :param url: original url
        :return:    truncated url
        """
        return (url[:50] + url[-50:]) if len(url) > 100 else url
