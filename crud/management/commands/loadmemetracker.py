# -*- coding: utf-8 -*-
import json
import os
import traceback
from django.conf import settings
from django.core.management.base import BaseCommand
import time
from memetracker import models as meme_models
from crud.models import *


class Command(BaseCommand):
    help = 'Copy data from "memetracker" db into the default db'

    def handle(self, *args, **options):
        try:
            start = time.time()

            dir_path = os.path.join(settings.BASEPATH, 'data', 'memetracker')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            paths = {var: os.path.join(dir_path, '%s.json' % var) for var in ['nets', 'users', 'posts', 'memes']}

            if os.path.exists(paths['nets']):
                net_map = json.load(open(paths['nets'], 'r'))
                net_map = {int(key): value for key, value in net_map.items()}
            else:
                net_map = {}
                for net in meme_models.SocialNet.objects.all():
                    new_net = SocialNet.objects.create(name=net.name, icon=net.icon)
                    net_map[net.id] = new_net.id
                json.dump(net_map, open(paths['nets'], 'w'), indent=4)

            if os.path.exists(paths['users']):
                user_map = json.load(open(paths['users'], 'r'))
                user_map = {int(key): value for key, value in user_map.items()}
            else:
                self.stdout.write('creating users ...')
                user_map = {}
                for user in meme_models.UserAccount.objects.all():
                    new_user = UserAccount.objects.create(social_net_id=net_map[user.social_net_id],
                                                          username=user.username, avatar=user.avatar)
                    user_map[user.id] = new_user.id
                json.dump(user_map, open(paths['users'], 'w'), indent=4)

            if os.path.exists(paths['posts']):
                post_map = json.load(open(paths['posts'], 'r'))
                post_map = {int(key): value for key, value in post_map.items()}
            else:
                self.stdout.write('creating posts ...')
                post_map = {}
                for post in meme_models.Post.objects.all():
                    new_post = Post.objects.create(author_id=user_map[post.author_id], datetime=post.datetime,
                                                   text=post.text, url=post.url)
                    post_map[post.id] = new_post.id
                json.dump(post_map, open(paths['posts'], 'w'), indent=4)

            self.stdout.write('creating reshares ...')
            step = 100000
            count = 0
            reshares = []
            for resh in meme_models.Reshare.objects.all():
                reshares.append(
                    Reshare(post_id=post_map[resh.post_id], reshared_post_id=post_map[resh.reshared_post_id],
                            user_id=user_map[resh.user_id], ref_user_id=user_map[resh.ref_user_id],
                            datetime=resh.datetime, ref_datetime=resh.ref_datetime)
                )
                count += 1
                if count % step == 0:
                    Reshare.objects.bulk_create(reshares)
                    reshares = []

            self.stdout.write('creating diffusionparams ...')
            step = 100000
            count = 0
            params = []
            for obj in meme_models.DiffusionParam.objects.all():
                params.append(
                    DiffusionParam(sender_id=user_map[obj.sender_id], receiver_id=user_map[obj.receiver_id],
                                   weight=obj.weight, delay=obj.delay))
                count += 1
                if count % step == 0:
                    DiffusionParam.objects.bulk_create(params)
                    params = []

            del user_map

            if os.path.exists(paths['memes']):
                meme_map = json.load(open(paths['memes'], 'r'))
                meme_map = {int(key): value for key, value in meme_map.items()}
            else:
                self.stdout.write('creating memes ...')
                meme_map = {}
                for meme in meme_models.Meme.objects.all():
                    new_meme = Meme.objects.create(text=meme.text, count=meme.count, first_time=meme.first_time,
                                                   last_time=meme.last_time)
                    meme_map[meme.id] = new_meme.id
                json.dump(meme_map, open(paths['memes'], 'w'), indent=4)

            self.stdout.write('creating postmemes ...')
            step = 100000
            count = 0
            postmemes = []
            for obj in meme_models.PostMeme.objects.all():
                postmemes.append(PostMeme(post_id=post_map[obj.post_id], meme_id=meme_map[obj.meme_id]))
                count += 1
                if count % step == 0:
                    PostMeme.objects.bulk_create(postmemes)
                    postmemes = []

            self.stdout.write('command done in %.2f min' % ((time.time() - start) / 60.0))
        except:
            self.stdout.write(traceback.format_exc())
            raise
