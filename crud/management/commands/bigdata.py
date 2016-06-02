# -*- coding: utf-8 -*-
from os import listdir
import os
import random
import re
from datetime import datetime

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.utils.timezone import get_default_timezone_name
import numpy as np
import pytz
from crud.models import *

NETS = [
    {
        "name": "googleplus",
        "icon": "/media/img/googleplus.png"
    },
    {
        "name": "facebook",
        "icon": "/media/img/facebook.jpg"
    },
    {
        "name": "twitter",
        "icon": "/media/img/twitter.png"
    }
]

MAIN_NET = 'facebook'

AVATARS_PATH = os.path.join(settings.MEDIA_ROOT, 'avatars')

DATETIME_STD_DEV = 86400  # Standard deviation = 1 day
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

TOPICS = ['business', 'entertainment', 'politics', 'sport', 'tech']

MULTI_ACC_CENTERS = 5
MULTI_ACC_DEPTH = 2


class Command(BaseCommand):
    help = 'Generates big data using data set'

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        SocialNet.objects.all().delete()
        Friendship.objects.all().delete()
        UserAccount.objects.all().delete()
        Post.objects.all().delete()
        Like.objects.all().delete()
        Comment.objects.all().delete()
        Reshare.objects.all().delete()
        Person.objects.all().delete()

        self.create_nets()
        user_id_map = self.create_friends()
        self.create_post_li_com_re(user_id_map)
        self.set_user_info()
        self.set_post_texts()
        #self.create_multi_accounts()

        self.stdout.write('Successfully generated big data')

    def create_nets(self):
        for obj in NETS:
            SocialNet.objects.create(name=obj['name'], icon=obj['icon'])

    def create_friends(self):
        net = SocialNet.objects.get(name=MAIN_NET)
        user_id_map = {}

        with open(os.path.join(settings.WOSN_PATH, 'facebook-links.txt.anon')) as f:
            count = 0
            while True:
                line = f.readline()
                if not line:
                    break
                m = re.match(r"(\d+)\s+(\d+)\s+(\S+)", line)
                user1_id, user2_id, timestamp = m.groups()
                if timestamp == '\\N':
                    dt = None
                else:
                    dt = datetime.fromtimestamp(int(timestamp))
                    dt = pytz.timezone(get_default_timezone_name()).localize(dt)

                if user1_id in user_id_map:
                    user1 = user_id_map[user1_id]
                else:
                    user1 = UserAccount.objects.create(social_net=net)
                    user_id_map[user1_id] = user1

                if user2_id in user_id_map:
                    user2 = user_id_map[user2_id]
                else:
                    user2 = UserAccount.objects.create(social_net=net)
                    user_id_map[user2_id] = user2

                Friendship.objects.create(user1=user1, user2=user2, start_datetime=dt)

                count += 1
                if count % 100 == 0:
                    self.stdout.write('%d friendships created' % count)

        return user_id_map

    def create_post_li_com_re(self, user_id_map):
        net = SocialNet.objects.get(name=MAIN_NET)

        with open(os.path.join(settings.WOSN_PATH, 'facebook-wall.txt.anon')) as f:
            count = 0
            while True:
                line = f.readline()
                if not line:
                    break
                m = re.match(r"(\d+)\s+(\d+)\s+(\d+)", line)
                user2_id, user1_id, timestamp = m.groups()

                timestamp = int(timestamp)
                p_dt = datetime.fromtimestamp(int(timestamp))
                l_dt1 = datetime.fromtimestamp(int(timestamp + abs(random.gauss(0, DATETIME_STD_DEV))))
                l_dt2 = datetime.fromtimestamp(int(timestamp + abs(random.gauss(0, DATETIME_STD_DEV))))
                c_dt1 = datetime.fromtimestamp(int(timestamp + abs(random.gauss(0, DATETIME_STD_DEV))))
                c_dt2 = datetime.fromtimestamp(int(timestamp + abs(random.gauss(0, DATETIME_STD_DEV))))
                r_dt = datetime.fromtimestamp(int(timestamp + abs(random.gauss(0, DATETIME_STD_DEV))))
                (p_dt, l_dt1, l_dt2, c_dt1, c_dt2, r_dt) = (pytz.timezone(get_default_timezone_name()).localize(dt) for
                                                            dt in [p_dt, l_dt1, l_dt2, c_dt1, c_dt2, r_dt])

                if user1_id in user_id_map:
                    user1 = user_id_map[user1_id]
                else:
                    user1 = UserAccount.objects.create(social_net=net)
                    user_id_map[user1_id] = user1

                if user2_id in user_id_map:
                    user2 = user_id_map[user2_id]
                else:
                    user2 = UserAccount.objects.create(social_net=net)
                    user_id_map[user2_id] = user2

                post = Post.objects.create(author=user1, datetime=p_dt)
                for dt in (l_dt1, l_dt2):
                    Like.objects.create(user=user2, post=post, datetime=dt)
                for dt in (c_dt1, c_dt2):
                    Comment.objects.create(user=user2, post=post, datetime=dt)
                Reshare.objects.create(user=user2, post=post, datetime=r_dt)

                count += 1
                if count % 100 == 0:
                    self.stdout.write('posts, likes, comments, reshares created for %d datetimes' % count)

    def set_user_info(self):
        with open(settings.MALE_FNAME) as f:
            content = f.read()
            male_names = content.strip().split('\n')
        with open(settings.FEMALE_FNAME) as f:
            content = f.read()
            female_names = content.strip().split('\n')
        avatars = os.listdir(AVATARS_PATH)

        count = 0

        for user in UserAccount.objects.all():
            person = Person()
            if random.random() < 0.5:
                person.gender = True  # male
                person.first_name = random.choice(male_names)
                person.last_name = random.choice(male_names)
            else:
                person.gender = False  # female
                person.first_name = random.choice(female_names)
                person.last_name = random.choice(female_names)
            person.language = random.choice(LANGUAGES)[0]
            person.save()
            user.person = person
            user.username = '%s_%s' % (person.first_name[0].lower(), person.last_name.lower())
            user.avatar = '/media/avatars/%s' % random.choice(avatars)
            user.save()

            count += 1
            if count % 100 == 0:
                self.stdout.write('%d persons created' % count)

    ##################### Set Post Texts #####################

    def set_post_texts(self):
        users = UserAccount.objects.all()

        # Sample center user topics randomly to one of the topics.
        self.stdout.write('center user topics:')
        centers_count = len(TOPICS)
        center_ids = []
        center_topics = np.zeros([len(TOPICS), len(TOPICS)])
        for i in range(len(TOPICS)):
            user_id = random.choice(users).id
            center_ids.append(user_id)
            vec = np.zeros(len(TOPICS))
            vec[i] = 1
            center_topics[i, :] = vec

            topic = Topic.objects.create(name=TOPICS[i])
            topic.users.add(UserAccount.objects.get(id=user_id))
            self.stdout.write('"%s" : "%s",' % (center_ids[i], TOPICS[i]))

        # Initialize distances by -1 and center distances to themselves by 0.
        distances = {}
        for user in users:
            distances[user.id] = -np.ones(centers_count)
        for i in range(centers_count):
            distances[center_ids[i]][i] = 0

        # Calculate distances to centers by BFS.
        for i in range(centers_count):

            print '>>>> calculating distances to center %d' % i
            queue = [center_ids[i]]
            count = 1
            depth = 1

            while len(queue) > 0:
                friendships = Friendship.objects.filter(Q(user1__id__in=queue) | Q(user2__id__in=queue))
                friend_ids = []
                for fr in friendships:
                    if distances[fr.user1.id][i] == -1:
                        fid = fr.user1.id
                    elif distances[fr.user2.id][i] == -1:
                        fid = fr.user2.id
                    else:
                        continue
                    distances[fid][i] = depth
                    friend_ids.append(fid)

                count += len(friend_ids)
                print 'center %d, depth %d : %d users done' % (i, depth, count)
                depth += 1
                queue = friend_ids

        count = 0

        for user in users:
            # Calculate topic vector according to distances to centers.
            if user.id in center_ids:
                vec = center_topics[center_ids.index(user.id), :]
            else:
                nz_distances = distances[user.id] + 1
                indexes = np.nonzero(nz_distances)
                nz_distances = nz_distances[indexes]
                weights = np.divide(1, nz_distances)
                nz_cent_topics = center_topics[indexes[0], :]

                try:
                    vec = np.average(nz_cent_topics, axis=0, weights=weights)
                except ZeroDivisionError:
                    continue
                vec += np.random.normal(0, 0.05, len(TOPICS))
                vec = np.fabs(vec)
                vec /= np.linalg.norm(vec)

            # Sample post texts based on the topic vector.
            for post in Post.objects.filter(author=user).all():
                topic = TOPICS[self.sample_by_dist(vec)]
                text = self.sample_text(topic)
                post.text = text
                count += 1
                if count % 100 == 0:
                    self.stdout.write('%d post texts sampled' % count)

    def sample_by_dist(self, distribution):
        s = random.random()
        cdf = 0
        for i in range(len(distribution)):
            if s <= cdf + distribution[i]:
                return i
            else:
                cdf += distribution[i]

    def sample_text(self, topic):
        path = os.path.join(settings.NEWS_PATH, topic)
        fname = random.choice(listdir(path))
        with open(os.path.join(path, fname)) as f:
            return f.read()

    ##################### Create Multiple Accounts #####################

    def create_multi_accounts(self):
        other_nets = list(set([net['name'] for net in NETS]) - {MAIN_NET})

        users = UserAccount.objects.all()
        centers = [random.choice(users) for i in range(MULTI_ACC_CENTERS)]

        for net_name in other_nets:
            net = SocialNet.objects.get(name=net_name)
            for user in centers:
                neighbors = self.get_depth_neighbors(user, MULTI_ACC_DEPTH)
                self.create_sub_net(neighbors, net)

    def get_depth_neighbors(self, root_user, depth):
        queue = [root_user]
        whole_set = {root_user}

        for d in range(depth):
            friendships = Friendship.objects.filter(Q(user1__in=queue) | Q(user2__in=queue))
            fr_users = set()
            for fr in friendships:
                fr_users.update([fr.user1, fr.user2])
            fr_users -= whole_set
            queue = list(fr_users)
            whole_set.update(queue)
            print 'depth %d done' % (d + 1)

        return list(whole_set)

    def create_sub_net(self, users, net):
        new_id_map = {}
        single_users = []
        new_users = []

        for main_user in users:
            person = main_user.person
            users = UserAccount.objects.filter(social_net=net, person=person)
            if not users:
                avatars = os.listdir(AVATARS_PATH)
                avatar_url = '/media/avatars/%s' % random.choice(avatars)
                username = '%s_%s_%s' % (person.first_name[0].lower(), person.last_name.lower(), net.name[0].lower())
                user = UserAccount.objects.create(social_net=net, person=person, avatar=avatar_url, username=username)
                new_users.append(user)
                single_users.append(main_user)
            else:
                user = users[0]
            new_id_map[main_user.id] = user.id

        self.stdout.write('%s: %d nodes created' % (net.name, len(new_id_map)))

        friendships = Friendship.objects.filter(user1__in=users, user2__in=users)
        links_count = int(len(friendships) / 2)
        indexes = np.random.choice(len(friendships), links_count)

        for i in indexes:
            user1 = new_id_map[friendships[i].user1]
            user2 = new_id_map[friendships[i].user2]
            Friendship.objects.create(user1=user1, user2=user2)

        self.stdout.write('%s: %d links created' % (net['name'], links_count))
