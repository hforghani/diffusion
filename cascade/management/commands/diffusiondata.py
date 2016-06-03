# -*- coding: utf-8 -*-
from optparse import make_option
import re
import traceback
from datetime import timedelta
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.core.management.base import BaseCommand
from django.db.models.aggregates import Count, Min, Max
import time
from crud.models import UserAccount, SocialNet, Post, Reshare, Meme, PostMeme
from utils.time_utils import str_to_datetime


class Command(BaseCommand):
    help = 'Create database entities using memetracker dataset.'

    option_list = BaseCommand.option_list + (
        make_option(
            "-s",
            "--start",
            type="int",
            dest="start_index",
            help="determine which index of post in the dataset file to start from"
        ),
        make_option(
            "-a",
            "--attributes",
            action="store_true",
            dest="set_attributes",
            help="just set attributes and ignore creating data"
        ),
        make_option(
            "-c",
            "--clear",
            action="store_true",
            dest="clear",
            help="clear existing data and continue"
        ),
        make_option(
            "-e",
            "--entities",
            action="store_true",
            dest="entities",
            help="just create meme, user_account, and post entities"
        ),
        make_option(
            "-r",
            "--relations",
            action="store_true",
            dest="relations",
            help="just create post_meme and reshare relations"
        ),
        make_option(
            "-t",
            "--text",
            action="store_true",
            dest="post_texts",
            help="Set post texts according to their memes"
        ),
    )

    def __init__(self):
        super(Command, self).__init__()
        self.memes_map = {}  # Map from meme texts to ids

    def handle(self, *args, **options):
        try:
            start = time.time()

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
                self.stdout.write('correcting remained posts ...')
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

                # Create social net.
                try:
                    net = SocialNet.objects.get(name='memetracker')
                except ObjectDoesNotExist:
                    net = SocialNet.objects.create(name='memetracker', icon='/media/img/memetracker.gif')

                path = settings.MEMETRACKER_PATH

                # Create instances of non-relation entities.
                if not (options['entities'] or options['relations']) or options['entities']:
                    self.stdout.write('======== creating entities ...')
                    self.create_entities(path, net)

                # Create instances of relation entities.
                if not (options['entities'] or options['relations']) or options['relations']:
                    self.load_memes()
                    self.stdout.write('======== creating relations ...')
                    self.create_relations(path, start_index=options['start_index'])

            # Set the meme count, first time, and last time attributes of memes.
            self.stdout.write('======== setting counts and publication times for the memes ...')
            self.calc_memes_values()

            self.stdout.write('======== command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise

    def create_entities(self, path, net):
        urls = []
        memes = []
        i = 0

        self.stdout.write('reading urls and memes from dataset ...')
        with open(path) as f:
            line = f.readline()

            while line:
                char = line[0]
                text = line[2:-1]
                if char in 'PL':
                    urls.append(self.truncate_url(text))
                elif char == 'Q':
                    memes.append(text)
                if char == 'P':
                    i += 1
                    if i % 1000000 == 0:
                        print '%d posts read' % i
                line = f.readline()
        print '%d posts read' % i

        self.stdout.write('loading existing memes ...')
        existing_memes = set(Meme.objects.values_list('text', flat=True))
        memes = set(memes) - existing_memes
        del existing_memes
        self.stdout.write('creating %d new memes ...' % len(memes))
        meme_entities = []
        i = 0
        for text in memes:
            meme_entities.append(Meme(text=text))
            i += 1
            if i % 1000000 == 0:
                Meme.objects.bulk_create(meme_entities)
                self.stdout.write('%d memes created' % i)
                meme_entities = []
        Meme.objects.bulk_create(meme_entities)
        self.stdout.write('%d memes created' % len(meme_entities))
        del memes
        del meme_entities

        self.stdout.write('loading existing urls ...')
        existing_urls = set(Post.objects.values_list('url', flat=True))
        urls = set(urls) - existing_urls
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

        self.stdout.write('creating %d new posts ...' % len(urls))
        posts = []
        i = 0
        for url in urls:
            username = self.get_username(url)
            if username:
                user_id = users_map[username]
                posts.append(Post(url=url, text='', author_id=user_id))
            i += 1
            if i % 1000 == 0:
                Post.objects.bulk_create(posts)
                print '%d posts created' % i
                posts = []
        Post.objects.bulk_create(posts)
        print '%d posts created' % len(posts)
        del posts, urls, users_map

    def create_relations(self, path, start_index):
        source_urls = []
        post_url = None
        datetime = None
        meme_ids = []
        meme_texts = []
        post_memes = []
        reshares = []
        i = 0
        t0 = time.time()

        # if 'start_index' is specified, ignore lower indexes.
        ignoring = False
        if start_index:
            ignoring = True

        with open(path) as f:
            line = f.readline()

            while line:
                char = line[0]
                text = line[2:-1]

                # Count posts.
                if char == 'P':
                    i += 1
                    if (not ignoring or i == start_index) and i % 100 == 0:
                        self.stdout.write('processing post %d' % i)
                    elif i % 10000 == 0:
                        self.stdout.write('ignoring post %d' % i)

                # Handle if it is in ignoring state.
                if ignoring:
                    if i < start_index:
                        line = f.readline()
                        continue
                    else:
                        ignoring = False
                        t0 = time.time()

                if char == 'P':  # post line
                    if post_url is not None:
                        pm, resh = self.get_post_rels(post_url, datetime, meme_ids, meme_texts, source_urls)
                        post_memes.extend(pm)
                        reshares.extend(resh)
                        if i % 5000 == 0:
                            self.stdout.write(
                                'saving %d post memes and %d reshares ...' % (len(post_memes), len(reshares)))
                            PostMeme.objects.bulk_create(post_memes)
                            Reshare.objects.bulk_create(reshares)
                            post_memes = []
                            reshares = []
                            self.stdout.write('time : %f' % (time.time() - t0))
                            t0 = time.time()

                    post_url = text
                    source_urls = []
                    meme_ids = []
                    meme_texts = []
                elif char == 'T':  # time line
                    datetime = str_to_datetime(text)
                elif char == 'Q':  # meme line
                    meme_id = self.memes_map[text]
                    meme_ids.append(meme_id)
                    meme_texts.append(text)
                elif char == 'L':  # link line
                    source_urls.append(text)

                line = f.readline()

        pm, resh = self.get_post_rels(post_url, datetime, meme_ids, meme_texts, source_urls)
        self.stdout.write('saving post memes and reshares ...')
        post_memes.extend(pm)
        reshares.extend(resh)
        PostMeme.objects.bulk_create(post_memes)
        Reshare.objects.bulk_create(reshares)

    def get_post_rels(self, post_url, datetime, meme_ids, meme_texts, source_urls):
        """
        Assign post memes, and create reshares for the referenced links.
        """
        # Create the post.
        post = Post.objects.filter(url=self.truncate_url(post_url))[0]
        post.datetime = datetime
        post.text = '. '.join(meme_texts)
        post.save()

        # Assign the memes to the post.
        post_memes = [PostMeme(post_id=post.id, meme_id=mid) for mid in meme_ids]

        # Create reshares if the post is reshared.
        reshares = []
        for src_url in set(source_urls) - {post_url}:
            try:
                src_post = Post.objects.filter(url=self.truncate_url(src_url))[0]
            except IndexError:
                src_post = None
            if src_post:
                # If the link does not have datetime means it is a line staring by 'L' for the first time.
                # So assign the same memes as the post.
                if not src_post.datetime:
                    post_memes.extend([PostMeme(post=src_post, meme_id=mid) for mid in meme_ids])
                if not src_post.datetime or datetime - timedelta(days=1) < src_post.datetime:
                    src_post.datetime = datetime - timedelta(days=1)
                    if not src_post.text or not src_post.text.strip():
                        cur_memes = set()
                    else:
                        cur_memes = set(src_post.text.split('. '))
                    cur_memes.update(meme_texts)
                    src_post.text = '. '.join(cur_memes)
                    src_post.save()
                reshares.append(Reshare(post_id=post.id, reshared_post=src_post, datetime=datetime))

        return post_memes, reshares

    def load_memes(self):
        self.stdout.write('loading meme ids ...')
        memes = Meme.objects.values('text', 'id')
        self.memes_map = {meme['text']: meme['id'] for meme in memes}
        del memes

    def calc_memes_values(self):
        self.stdout.write('getting meme ids ...')
        meme_ids = list(Meme.objects.values_list('id', flat=True))
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
        for mid in meme_ids:
            try:
                count = meme_counts[mid]
            except:
                count = None
            try:
                f_time = first_times[mid]
            except:
                f_time = None
            try:
                l_time = last_times[mid]
            except:
                l_time = None
            Meme.objects.filter(id=mid).update(count=count, first_time=f_time, last_time=l_time)
            i += 1
            if i % 10000 == 0:
                self.stdout.write('%d memes saved' % i)

    def get_username(self, url):
        try:
            return re.match(r'https?://([^/?]+)', url.lower()).groups()[0][:100]
        except AttributeError:
            return None

    def truncate_url(self, url):
        trunc = (url[:50] + url[-50:]) if len(url) > 100 else url
        try:
            return trunc.decode('utf-8')
        except UnicodeDecodeError:
            return trunc.decode('latin-1')
