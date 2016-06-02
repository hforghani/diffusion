# -*- coding: utf-8 -*-
import random
from django.core.management.base import BaseCommand
from crud.models import UserAccount, Friendship


class Command(BaseCommand):
    help = 'Generate random friendships'

    def handle(self, *args, **options):
        FRIENDSHIP_COUNT = 100000
        friendships = []
        users_count = UserAccount.objects.count()
        users = UserAccount.objects.all()
        friends = list(Friendship.objects.values_list('user1', 'user2'))

        for i in range(FRIENDSHIP_COUNT):
            i1 = random.randint(0, users_count - 1)
            i2 = random.randint(0, users_count - 1)
            user1 = users[i1]
            user2 = users[i2]
            if user1.id != user2.id and (user1.id, user2.id) not in friends and (user2.id, user1.id) not in friends:
                friendships.append(Friendship(user1=user1, user2=user2))
                friends.append((user1.id, user2))
            if (i + 1) % 100 == 0:
                self.stdout.write('%d done' % (i + 1))
        self.stdout.write('creating bulk ...')
        Friendship.objects.bulk_create(friendships)
