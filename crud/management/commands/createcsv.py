# -*- coding: utf-8 -*-
import datetime
from optparse import make_option
import os
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
import unicodecsv as csv
import traceback
from django.core.management.base import BaseCommand
from crud.models import UserAccount, Friendship, SocialNet, Comment, Post, Like, Reshare
from utils.time_utils import DT_FORMAT


class Command(BaseCommand):
    help = 'Write a existing data to csv files'

    def handle(self, *args, **options):
        try:
            net_name = args[0]
            try:
                SocialNet.objects.get(name=net_name)
            except ObjectDoesNotExist:
                raise Exception('Social net "%s" does not exist' % net_name)
            data_path = os.path.join(settings.DATA_ENTRY_PATH, net_name)
            if not os.path.exists(data_path):
                os.mkdir(data_path)
                
            self.stdout.write('creating social net ...')
            self.write_csv(os.path.join(data_path, 'nets.csv'), SocialNet, ['id', 'name', 'icon'],
                           SocialNet.objects.filter(name=net_name))
            self.stdout.write('creating users ...')
            self.write_csv(os.path.join(data_path, 'users.csv'), UserAccount,
                           ['id', 'social_net_id', 'username', 'person__first_name', 'person__last_name',
                            'person__gender', 'person__birth_loc', 'person__location', 'person__language',
                            'person__bio', 'start_datetime', 'exit_datetime', 'avatar'])
            self.stdout.write('creating friendships ...')
            self.write_csv(os.path.join(data_path, 'friendships.csv'), Friendship,
                           ['user1_id', 'user2_id', 'start_datetime', 'end_datetime'])
            self.stdout.write('creating posts ...')
            self.write_csv(os.path.join(data_path, 'posts.csv'), Post, ['id', 'author_id', 'datetime', 'text'])
            self.stdout.write('creating comments ...')
            self.write_csv(os.path.join(data_path, 'comments.csv'), Comment, ['user_id', 'post_id', 'datetime', 'text'])
            self.stdout.write('creating likes ...')
            self.write_csv(os.path.join(data_path, 'likes.csv'), Like, ['user_id', 'post_id', 'datetime'])
            self.stdout.write('creating reshares ...')
            self.write_csv(os.path.join(data_path, 'reshares.csv'), Reshare, ['post_id', 'reshared_post_id'])
        except:
            self.stdout.write(traceback.format_exc())
            raise

    def write_csv(self, filename, model_class, fields, queryset=None):
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if queryset:
                iterator = queryset.values_list(*fields).iterator()
            else:
                iterator = model_class.objects.values_list(*fields).iterator()
            for obj in iterator:
                obj = list(obj)
                for i in range(len(obj)):
                    if isinstance(obj[i], datetime.datetime):
                        obj[i] = obj[i].strftime(DT_FORMAT)
                writer.writerow([unicode(field) if field is not None else '' for field in obj])
