# -*- coding: utf-8 -*-
import json
from optparse import make_option
import os
import traceback
from django.conf import settings
from django.db.models import Count
import numpy as np
from django.core.management.base import BaseCommand
import time
from scipy import sparse
from crud.models import Meme, UserAccount, Reshare, Post


class Command(BaseCommand):
    help = ''

    option_list = BaseCommand.option_list + (
        make_option(
            "-n",
            "--number",
            type="int",
            dest="train_num",
            help="number of training samples",
        ),
        make_option(
            "-o",
            "--output",
            type="string",
            dest="out_file",
            help="path of output file",
        ),
    )

    def __init__(self):
        super(Command, self).__init__()
        self._follow_thr = 0.1

    def handle(self, *args, **options):
        try:
            start = time.time()

            train_set_path = os.path.join(settings.BASEPATH, 'data', 'train_set.json')
            if os.path.exists(train_set_path):
                train_memes = json.load(open(train_set_path))
            else:
                if options['train_num']:
                    train_count = options['train_num']
                else:
                    train_count = 2.0 / 3 * Meme.objects.filter(depth__gte=1).count()
                train_memes = list(
                    np.random.choice(Meme.objects.filter(depth__gte=1).values_list('id', flat=True), train_count,
                                     replace=False))
            self.stdout.write('rules for %d training memes will be created' % len(train_memes))

            if options['out_file']:
                out_file = options['out_file']
            else:
                out_file = os.path.join(settings.BASEPATH, 'data', 'rules.db')

            users_count = UserAccount.objects.count()
            user_ids = UserAccount.objects.values_list('id', flat=True)
            user_indexes = {user_ids[i]: i for i in range(len(user_ids))}

            self.write_follows(user_ids, user_indexes, train_memes, out_file)

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))

        except:
            self.stdout.write(traceback.format_exc())
            raise

    def write_follows(self, user_ids, user_indexes, train_memes, out_file):
        post_ids = Post.objects.filter(postmeme__meme_id__in=train_memes).values_list('id', flat=True).distinct()
        reshares = Reshare.objects.filter(post__in=post_ids, reshared_post__in=post_ids)

        self.stdout.write('counting total reshares ...')
        users_count = len(user_ids)
        total_resh_count = np.zeros(users_count)
        for resh in reshares.values('user_id').annotate(count=Count('id')):
            total_resh_count[user_indexes[resh['user_id']]] = resh['count']

        self.stdout.write('counting pairwise reshares ...')
        resh_count = sparse.lil_matrix((users_count, users_count))
        for resh in reshares.values('ref_user_id', 'user_id').annotate(count=Count('id')):
            u1 = user_indexes[resh['ref_user_id']]
            u2 = user_indexes[resh['user_id']]
            resh_count[u1, u2] = resh['count']

        self.stdout.write('writing followship rules ...')
        with open(out_file, 'w') as f:
            rows, cols = resh_count.nonzero()
            for i in range(len(rows)):
                u1, u2 = rows[i], cols[i]
                ratio = float(resh_count[u1, u2]) / total_resh_count[u2]
                if ratio > self._follow_thr:
                    f.write('follows(u%d, u%d)\n' % (user_ids[u2], user_ids[u1]))
