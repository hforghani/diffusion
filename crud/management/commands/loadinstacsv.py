# -*- coding: utf-8 -*-
import csv
import json
from optparse import make_option
import os
import traceback
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.core.management.base import BaseCommand
import time
from crud.models import *
from utils.time_utils import localize, str_to_datetime


class Command(BaseCommand):
    help = 'Copy instagram data from "insta" db csv files into the default db'

    option_list = BaseCommand.option_list + (
        make_option("-d", "--dir", type="string", dest="dir", help="path of the directory containing csv files"),
        make_option("-0", "--level0", type="string", dest="level0",
                    help="set the name of the project of users in level 0; do not create the project otherwise"),
        make_option("-u", "--users", action="store_true", dest="users",
                    help="create users and level0 project if wanted"),
        make_option("-p", "--posts", action="store_true", dest="posts", help="create posts"),
        make_option("-c", "--comments", action="store_true", dest="comments", help="create comments"),
        make_option("-l", "--likes", action="store_true", dest="likes", help="create likes"),
    )

    def handle(self, *args, **options):
        try:
            start = time.time()
            no_options = True
            for opt in ['users', 'posts', 'comments', 'likes']:
                if opt in options:
                    no_options = False
                    break

            if 'dir' in options:
                path = options['dir']
            else:
                raise Exception('no directory given')

            dir_path = os.path.join(settings.BASEPATH, 'resources', 'insta')
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            save_paths = {var: os.path.join(dir_path, '%s.json' % var) for var in ['users', 'posts', 'tags']}

            if not SocialNet.objects.exists():
                net = SocialNet.objects.create(name='instagram', icon='/media/img/insta.ico')
            else:
                net = SocialNet.objects.get(name='instagram')

            if os.path.exists(save_paths['users']):
                users = json.load(open(save_paths['users'], 'r'))
            elif options['users'] or no_options:
                self.stdout.write('creating person\'s and users ...')
                users = {}
                t0 = time.time()
                level0 = []

                with open(os.path.join(path, 'user.csv'), 'rb') as csvfile:
                    insta_users = csv.reader(csvfile, delimiter='\t')
                    for row in insta_users:
                        uid, bio, _, first_name, _, _, _, last_name, _, profile_picture, _, username, _, level = row[
                                                                                                                 :14]
                        bio = None if bio == '\\N' else bio.replace('\\n', '\n')
                        first_name = None if first_name == '\\N' else first_name
                        last_name = None if last_name == '\\N' else last_name
                        try:
                            user = UserAccount.objects.get(username=username)
                            users[uid] = user.id
                        except ObjectDoesNotExist:
                            person = Person.objects.create(first_name=first_name, last_name=last_name, bio=bio)
                            user = UserAccount.objects.create(social_net=net, person=person, username=username,
                                                              avatar=profile_picture)
                            users[uid] = user.id
                        if options['level0'] and level == 'ZERO':
                            level0.append(user.id)
                        if time.time() - t0 > 30:
                            self.stdout.write('%d users done' % len(users))
                            t0 = time.time()

                if options['level0']:
                    self.stdout.write('creating level0 project ...')
                    project = Project.objects.create(name=options['level0'], social_net=net)
                    for user_id in level0:
                        ProjectMembership.objects.create(project=project, user_id=user_id)
                json.dump(users, open(save_paths['users'], 'w'), indent=4)

            if os.path.exists(save_paths['posts']):
                posts = json.load(open(save_paths['posts'], 'r'))
            elif options['posts'] or no_options:
                self.stdout.write('collecting post texts ...')
                post_texts = {}
                t0 = time.time()
                with open(os.path.join(path, 'caption.csv'), 'rb') as csvfile:
                    captions = csv.reader(csvfile, delimiter='\t')
                    for row in captions:
                        _, _, _, text, _, post_id = row
                        text = None if text == '\\N' else text.replace('\\n', '\n')
                        post_texts[post_id] = text

                self.stdout.write('creating posts ...')
                posts = {}
                with open(os.path.join(path, 'media.csv'), 'rb') as csvfile:
                    insta_posts = csv.reader(csvfile, delimiter='\t')
                    for row in insta_posts:
                        pid, code, _, _, created_time, _, _, _, _, _, _, uid = row[:12]
                        if uid not in users:
                            continue
                        user = UserAccount.objects.get(id=users[uid])
                        if pid in post_texts:
                            text = post_texts[pid]
                        else:
                            text = None
                        url = 'https://www.instagram.com/p/%s/?taken-by=%s' % (code, user.username)
                        post = Post.objects.create(author=user, datetime=str_to_datetime(created_time), text=text,
                                                   url=url)
                        posts[pid] = post.id

                        if time.time() - t0 > 30:
                            self.stdout.write('%d posts done' % len(posts))
                            t0 = time.time()
                json.dump(posts, open(save_paths['posts'], 'w'), indent=4)

            if options['comments'] or no_options:
                self.stdout.write('creating comments ...')
                comments = []
                with open(os.path.join(path, 'comment.csv'), 'rb') as csvfile:
                    insta_comments = csv.reader(csvfile, delimiter='\t')
                    for row in insta_comments:
                        _, created_time, _, text, uid, pid = row
                        text = None if text == '\\N' else text.replace('\\n', '\n')
                        if uid not in users or pid not in posts:
                            continue
                        post_id = posts[pid]
                        user_id = users[uid]
                        comments.append(
                            Comment(post_id=post_id, user_id=user_id, datetime=str_to_datetime(created_time),
                                    text=text))
                Comment.objects.bulk_create(comments)

            if options['likes'] or no_options:
                self.stdout.write('creating likes ...')
                likes = []
                with open(os.path.join(path, 'like.csv'), 'rb') as csvfile:
                    insta_likes = csv.reader(csvfile, delimiter='\t')
                    for row in insta_likes:
                        _, pid, uid = row
                        if uid not in users or pid not in posts:
                            continue
                        post_id = posts[pid]
                        user_id = users[uid]
                        likes.append(Like(post_id=post_id, user_id=user_id))
                Like.objects.bulk_create(likes)

            self.stdout.write('command done in %.2f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
