# -*- coding: utf-8 -*-
import traceback
from django.core.management.base import BaseCommand
import time
from crud.models import Friendship, Like


class Command(BaseCommand):
    help = 'Extract friendships from likes'

    def handle(self, *args, **options):
        try:
            start = time.time()
            self.stdout.write('extracting friendships ...')
            objects = []
            likes = Like.objects.values('user_id', 'post__author_id').distinct()
            for like in likes.iterator():
                if not Friendship.objects.filter(user1_id=like['user_id'], user2_id=like['post__author_id']).exists():
                    objects.append(Friendship(user1_id=like['user_id'], user2_id=like['post__author_id']))
            Friendship.objects.bulk_create(objects)
            self.stdout.write('command done in %.2f min' % ((time.time() - start) / 60.0))
        except:
            self.stdout.write(traceback.format_exc())
            raise
