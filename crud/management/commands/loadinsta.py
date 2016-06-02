# -*- coding: utf-8 -*-
import json
import os
import traceback
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.core.management.base import BaseCommand
import time
from insta import models as insta_models
from crud.models import *
from utils.time_utils import localize


class Command(BaseCommand):
    help = 'Copy instagram data from "insta" db into the default db'

    def handle(self, *args, **options):
        try:
            start = time.time()

            dir_path = os.path.join(settings.BASEPATH, 'resources', 'insta')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            paths = {var: os.path.join(dir_path, '%s.json' % var) for var in ['users', 'posts', 'tags']}

            if not SocialNet.objects.exists():
                net = SocialNet.objects.create(name='instagram', icon='/media/img/insta.png')
            else:
                net = SocialNet.objects.get(name='instagram')

            if os.path.exists(paths['users']) and UserAccount.objects.exists():
                users = json.load(open(paths['users'], 'r'))
                users = {int(key): value for key, value in users.items()}
            else:
                self.stdout.write('creating persons and users ...')
                users = {}
                for user in insta_models.User.objects.all():
                    person = Person.objects.create(first_name=user.first_name, last_name=user.last_name, bio=user.bio)
                    user_account = UserAccount.objects.create(social_net=net, person=person, username=user.user_name,
                                                              avatar=user.profile_picture)
                    users[user.id] = user_account.id
                json.dump(users, open(paths['users'], 'w'), indent=4)

            if os.path.exists(paths['posts']) and Post.objects.exists():
                posts = json.load(open(paths['posts'], 'r'))
                posts = {int(key): value for key, value in posts.items()}
            else:
                self.stdout.write('creating posts ...')
                posts = {}
                for media in insta_models.Media.objects.all():
                    user = UserAccount.objects.get(id=users[media.user_id])
                    try:
                        text = insta_models.Caption.objects.get(media_id=media.id).text
                    except ObjectDoesNotExist:
                        text = None
                    url = 'https://www.instagram.com/p/%s/?taken-by=%s' % (media.code, user.username)
                    post = Post.objects.create(author_id=user.id, datetime=localize(media.created_time), text=text,
                                               url=url)
                    posts[media.id] = post.id
                json.dump(posts, open(paths['posts'], 'w'), indent=4)

            if os.path.exists(paths['tags']) and Hashtag.objects.exists():
                tags = json.load(open(paths['tags'], 'r'))
                tags = {int(key): value for key, value in tags.items()}
            else:
                self.stdout.write('creating tags ...')
                tags = {}
                for tag in insta_models.Tag.objects.all():
                    new_tag = Hashtag.objects.create(name=tag.name)
                    tags[tag.id] = new_tag.id
                json.dump(tags, open(paths['tags'], 'w'), indent=4)

            if not Comment.objects.exists():
                self.stdout.write('creating comments ...')
                comments = []
                for comment in insta_models.Comment.objects.all():
                    post_id = posts[comment.media_id]
                    user_id = users[comment.user.id]
                    comments.append(Comment(post_id=post_id, user_id=user_id, datetime=localize(comment.created_time),
                                            text=comment.text))
                Comment.objects.bulk_create(comments)

            if not Like.objects.exists():
                self.stdout.write('creating likes ...')
                likes = []
                for like in insta_models.Like.objects.all():
                    post_id = posts[like.media_id]
                    user_id = users[like.user_id]
                    likes.append(Like(post_id=post_id, user_id=user_id))
                Like.objects.bulk_create(likes)

            if not HashtagPost.objects.exists():
                self.stdout.write('creating post hashtags ...')
                hposts = []
                for hpost in insta_models.TagCaption.objects.all():
                    post_id = posts[hpost.caption.media_id]
                    tag_id = tags[hpost.tag_id]
                    hposts.append(HashtagPost(hashtag_id=tag_id, post_id=post_id))
                HashtagPost.objects.bulk_create(hposts)

            self.stdout.write('command done in %.2f min' % ((time.time() - start) / 60.0))
        except:
            self.stdout.write(traceback.format_exc())
            raise
