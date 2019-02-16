# -*- coding: utf-8 -*-
import os
import re
import traceback
from bson import SON
from bson.objectid import ObjectId

from django.core.management.base import BaseCommand, CommandError
import time
import pygtrie
from mongo import mongodb
from utils.time_utils import str_to_datetime


class Command(BaseCommand):
    help = 'Create database instances in MongoDB using memetracker dataset.'


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


    def handle(self, *args, **options):
        try:
            start = time.time()

            path = options['file']
            if not path:
                raise CommandError('file argument not specified')

            # Delete all data.
            if options['clear'] and not options['set_attributes']:
                self.stdout.write('======== deleting data ...')
                mongodb.postmemes.delete_many()
                mongodb.reshares.delete_many()
                mongodb.posts.delete_many()
                mongodb.memes.delete_many()
                mongodb.users.delete_many()
                mongodb.social_nets.delete_many()

            # Get or create social net.
            net = mongodb.social_nets.find_one({'name': 'memetracker'})
            if net is None:
                net_id = mongodb.social_nets.insert_one({'name': 'memetracker'}).inserted_id
                net = mongodb.social_nets.find_one({'_id': net_id})

            # Create instances of non-relation entities.
            if (options['entities'] or not options['relations']) and not options['set_attributes']:
                self.stdout.write('======== creating entities ...')
                self.create_entities(path, net)

            # Create instances of relation entities.
            if (options['relations'] or not options['entities']) and not options['set_attributes']:
                temp_data_path = os.path.join(os.path.dirname(path), os.path.basename(path) + '.temp')
                if not os.path.exists(temp_data_path):
                    self.stdout.write('creating temp file ...')
                    self.create_temp(path, temp_data_path)

                self.stdout.write('======== creating relations ...')
                self.create_relations(temp_data_path, start_index=options['start_index'])

            # Set the meme count, first time, and last time attributes of memes.
            if options['set_attributes']:
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
        existing_memes = {m['text'] for m in mongodb.memes.find({}, {'_id': 0, 'text': 1})}
        memes -= existing_memes
        del existing_memes
        self.stdout.write('creating %d new memes ...' % len(memes))
        meme_entities = []
        i = 0
        for text in memes:
            meme_entities.append({'text': text})
            i += 1
            if i % 100000 == 0:
                mongodb.memes.insert_many(meme_entities)
                self.stdout.write('%d memes created' % i)
                meme_entities = []

        if meme_entities:
            mongodb.memes.insert_many(meme_entities)
        self.stdout.write('%d memes created' % len(meme_entities))
        del memes
        del meme_entities

        self.stdout.write('loading existing urls ...')
        t0 = time.time()
        posts = mongodb.posts.find({}, {'_id': 0, 'url': 1})
        self.stdout.write('posts query done in {:.2f} m'.format((time.time() - t0) / 60))
        t0 = time.time()
        existing_urls = {p['url'] for p in posts}
        self.stdout.write('urls flatted in {:.2f} m'.format((time.time() - t0) / 60))
        t0 = time.time()
        urls = {key: value for key, value in urls.items() if key not in existing_urls}
        del existing_urls
        self.stdout.write('new urls extracted in {:.2f} m'.format((time.time() - t0) / 60))
        self.stdout.write('loading existing usernames ...')
        existing_usernames = {u['username'] for u in mongodb.users.find({}, {'_id': 0, 'username': 1})}
        usernames = set([self.get_username(url) for url in urls]) - existing_usernames - {None}
        del existing_usernames

        self.stdout.write('creating %d new users ...' % len(usernames))
        i = 0
        users = []
        for uname in usernames:
            users.append({'username': uname, 'social_net': net['_id']})
            i += 1
            if i % 100000 == 0:
                mongodb.users.insert_many(users)
                self.stdout.write('%d users created' % i)
                users = []

        if users:
            mongodb.users.insert_many(users)
        self.stdout.write('%d users created' % len(users))
        del usernames
        del users
        self.stdout.write('loading user ids ...')
        users = mongodb.users.find()
        users_map = {user['username']: user['_id'] for user in users}
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
                posts.append({'url': url, 'author_id': user_id, 'datetime': dt})
            else:
                try:
                    self.stdout.write("post with url '{}' ignored".format(url))
                except UnicodeEncodeError:
                    self.stdout.write("a post ignored having url with non-utf8 encoding")
            i += 1
            if i % 10000 == 0:
                mongodb.posts.insert_many(posts)
                posts = []
            if i % 100000 == 0:
                self.stdout.write('%d posts created (%.1f s)' % (i, (time.time() - t0)))
                t0 = time.time()

        if posts:
            mongodb.posts.insert_many(posts)
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

                # Count posts.
                if char == 'P':
                    i += 1
                    if (not ignoring or i == start_index) and i % 10000 == 0:
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

                text = line[2:-1]

                if char == 'P':  # post line
                    if post_id is not None:
                        pm, resh = self.get_post_rels(post_id, datetime, meme_ids, source_ids)
                        post_memes.extend(pm)
                        reshares.extend(resh)
                        if i % 10000 == 0:
                            self.stdout.write(
                                'saving %d post memes and %d reshares ...' % (len(post_memes), len(reshares)))
                            mongodb.postmemes.insert_many(post_memes)
                            mongodb.reshares.insert_many(reshares)
                            post_memes = []
                            reshares = []
                            self.stdout.write('time : %d s' % (time.time() - t0))
                            t0 = time.time()

                    if '/' not in text:
                        post_id = ObjectId(text)
                    else:
                        raise CommandError("invalid post id: '{}'".format(text))
                    source_ids = []
                    meme_ids = []
                elif char == 'T':  # time line
                    datetime = str_to_datetime(text)
                elif char == 'Q':  # meme line
                    if ' ' not in text:
                        meme_id = ObjectId(text)
                        meme_ids.append(meme_id)
                    else:
                        self.stdout.write("meme '{}' ignored".format(text))
                elif char == 'L':  # link line
                    if '/' not in text:
                        source_ids.append(ObjectId(text))
                    else:
                        try:
                            self.stdout.write("link '{}' ignored".format(text))
                        except UnicodeEncodeError:
                            self.stdout.write("a link with non-utf8 url ignored")

                line = f.readline()

        # Add the relations of the last post.
        pm, resh = self.get_post_rels(post_id, datetime, meme_ids, source_ids)
        post_memes.extend(pm)
        reshares.extend(resh)

        # Save the remaining relations.
        self.stdout.write(
            'saving %d post memes and %d reshares ...' % (len(post_memes), len(reshares)))
        mongodb.postmemes.insert_many(post_memes)
        mongodb.reshares.insert_many(reshares)

    # @profile
    def get_post_rels(self, post_id, datetime, meme_ids, source_ids):
        """
        Create PostMeme and Reshare instances for the referenced links. Just create the instances not inserting in the db.
        """
        # Create the post.
        post = mongodb.posts.find_one({'_id': post_id})
        if post is None:
            raise CommandError('post does not exist with id {}'.format(post_id))

        # Assign the memes to the post.
        post_memes = [{'post_id': ObjectId(post_id), 'meme_id': mid} for mid in meme_ids]

        # Create reshares if the post is reshared.
        reshares = []
        src_ids = set(source_ids) - {post_id}
        if src_ids:
            src_posts = mongodb.posts.find({'_id': {'$in': list(src_ids)}})
            if src_posts.count() != len(src_ids):  # Raise an error if some of link posts do not exist.
                not_existing = src_ids - {p['_id'] for p in src_posts}
                raise CommandError('link post does not exist with id(s): {}'.format(', '.join(not_existing)))
            for src_post in src_posts:
                reshares.append({'post_id': post['_id'], 'reshared_post_id': src_post['_id'], 'datetime': datetime,
                                 'user_id': post['author_id'], 'ref_user_id': src_post['author_id'],
                                 'ref_datetime': src_post['datetime']})

        return post_memes, reshares

    def create_temp(self, path, temp_path):
        # Replace meme texts with meme ids and create temporary data files.
        from_path = path
        memes_count = mongodb.memes.count()
        step = 10 ** 7
        i = 0
        t0 = time.time()
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
            self.stdout.write('done in %.2f min' % ((time.time() - t0) / 60))
            t0 = time.time()

        # Replace post urls with post ids and create temporary data files.
        posts_count = mongodb.posts.count()
        step = 10 ** 7
        i = 0
        t0 = time.time()
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
            self.stdout.write('done in %.2f min' % ((time.time() - t0) / 60))
            t0 = time.time()

        os.rename(from_path, temp_path)

    def replace(self, in_path, out_path, characters, replace_map):
        in_batch_size = 1000
        out_batch_size = 1000

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
        pipelines = [{'$sort': SON([('_id', 1)])},
                     {'$project': {'_id': 1, 'text': 1}}]
        if limit is not None or offset > 0:
            pipelines.append({'$skip': offset})
            if limit is not None:
                pipelines.append({'$limit': limit})
        memes = mongodb.memes.aggregate(pipelines)
        for meme in memes:
            memes_map[meme['text']] = meme['_id']
        return memes_map

    def load_posts(self, offset=0, limit=None):
        """
        Get map of post urls to post id's
        :return:
        """
        posts_map = pygtrie.StringTrie()  # A trie data structure that maps from meme texts to ids
        pipelines = [{'$sort': SON([('_id', 1)])},
                     {'$project': {'_id': 1, 'url': 1}}]
        if limit is not None or offset > 0:
            pipelines.append({'$skip': offset})
            if limit is not None:
                pipelines.append({'$limit': limit})

        posts = mongodb.posts.aggregate(pipelines)
        for post in posts:
            posts_map[post['url']] = post['_id']
        return posts_map

    def calc_memes_values(self):
        self.stdout.write('query of meme counts ...')
        meme_counts = mongodb.postmemes.aggregate([{'$group': {'_id': '$meme_id', 'count': {'$sum': 1}}}],
                                                  allowDiskUse=True)

        self.stdout.write('saving ...')
        for doc in meme_counts:
            mongodb.memes.find_one_and_update({'_id': doc['_id']}, {'$set': {'count': doc['count']}})
        del meme_counts

        self.stdout.write('query of first times ...')
        first_times = mongodb.postmemes.aggregate([
            {'$lookup': {'from': 'posts', 'localField': 'post_id',
                         'foreignField': '_id', 'as': 'post'}},
            {'$group': {'_id': '$meme_id', 'first': {'$min': '$post.datetime'}}},
            #{'$limit': 10000}
        ])

        self.stdout.write('saving ...')
        for doc in first_times:
            mongodb.memes.find_one_and_update({'_id': doc['_id']}, {'$set': {'first_time': doc['first']}})
        del first_times

        self.stdout.write('query of last times ...')
        last_times = mongodb.postmemes.aggregate([
            {'$lookup': {'from': 'posts', 'localField': 'post_id',
                         'foreignField': '_id', 'as': 'post'}},
            {'$group': {'_id': '$meme_id', 'last': {'$max': '$post.datetime'}}}
        ])

        self.stdout.write('saving ...')
        for doc in last_times:
            mongodb.memes.find_one_and_update({'_id': doc['_id']}, {'$set': {'last_time': doc['last']}})
        del last_times

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