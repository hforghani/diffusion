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
from cascade.models import CascadeTree, Project
from crud.models import Meme, UserAccount, Reshare, Post


class Command(BaseCommand):
    help = ''

    option_list = BaseCommand.option_list + (
        make_option(
            "-p",
            "--project",
            type="string",
            dest="project",
            help="project name",
        ),
        make_option(
            "-o", "--output",
            type="string",
            dest="out_file",
            help="path of output file",
        ),
        make_option(
            "-f",
            "--follows",
            action="store_true",
            dest="follows",
            help="just write 'follows' rules in the file"
        ),
        make_option(
            "-a",
            "--activates",
            action="store_true",
            dest="activates",
            help="just write 'activates' rules in the file"
        ),
        make_option(
            "-i",
            "--isactivated",
            action="store_true",
            dest="isactivated",
            help="just write 'isActivated' rules in the file"
        ),
        make_option(
            "-s",
            "--set",
            type="string",
            dest="set",
            default="train",
            help="create rules for training set or test set; valid values are 'train' or 'test'"
        ),
    )

    def __init__(self):
        super(Command, self).__init__()
        self._follow_thr = 0.1

    def handle(self, *args, **options):
        try:
            start = time.time()

            # Get project or raise exception.
            project_name = options['project']
            if project_name is None:
                raise Exception('project not specified')
            project = Project(project_name)

            # Load training and test sets and all cascade trees.
            train_memes, test_memes = project.load_data()
            trees = project.load_trees()

            # Get and delete the content of rules file.
            if options['out_file']:
                file_name = options['out_file']
            else:
                file_name = 'rules.db'
            out_file = os.path.join(project_path, file_name)
            open(out_file, 'w')

            do_all = not (options['follows'] or options['activates'] or options['isactivated'])

            users_count = UserAccount.objects.count()
            user_ids = UserAccount.objects.values_list('id', flat=True)
            user_indexes = {user_ids[i]: i for i in range(len(user_ids))}

            if options['set'] == 'train':
                self.stdout.write('rules will be created for training set')
                self.stdout.write('training set size = %d' % len(train_memes))
                if do_all or options['follows']:
                    self.stdout.write('>>> writing "follows" rules ...')
                    self.write_follows(user_ids, user_indexes, train_memes, out_file)

                if do_all or options['isactivated']:
                    self.stdout.write('>>> writing "isActivated" rules ...')
                    self.write_isactivated(trees, train_memes, out_file)

                if do_all or options['activates']:
                    self.stdout.write('>>> writing "activates" rules ...')
                    self.write_activates(trees, train_memes, out_file)

            elif options['set'] == 'test':
                self.stdout.write('rules will be created for test set')
                self.stdout.write('test set size = %d' % len(test_memes))
                if do_all or options['isactivated']:
                    self.stdout.write('>>> writing "isActivated" rules ...')
                    self.write_isactivated(trees, test_memes, out_file, initials=True)

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
                if u1 == u2:
                    continue
                ratio = float(resh_count[u1, u2]) / total_resh_count[u2]
                if ratio > self._follow_thr:
                    f.write('follows(u%d, u%d)\n' % (user_ids[u2], user_ids[u1]))
            f.write('\n')

    def write_isactivated(self, trees, meme_ids, out_file, initials=False):
        with open(out_file, 'a') as f:
            for meme_id in meme_ids:
                if initials:
                    nodes = trees[meme_id].tree
                else:
                    nodes = trees[meme_id].nodes()
                for node in nodes:
                    f.write('isActivated(u%d, m%d)\n' % (node.user_id, meme_id))
            f.write('\n')

    def write_activates(self, trees, meme_ids, out_file):
        with open(out_file, 'a') as f:
            for meme_id in meme_ids:
                edges = trees[meme_id].edges()
                for edge in edges:
                    f.write('activates(u%d, u%d, m%d)\n' % (edge[0], edge[1], meme_id))
            f.write('\n')
