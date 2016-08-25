# -*- coding: utf-8 -*-
import json
from optparse import make_option
import os
import traceback
from django.conf import settings
from django.core.management.base import BaseCommand
import time
from django.db.models import Q, Count
import numpy as np
from cascade.models import Project
from crud.models import Reshare, UserAccount, Post, Meme
from scipy import sparse


class Command(BaseCommand):
    help = 'Sample a subset of data and separate training and test sets and save them into the file'

    option_list = BaseCommand.option_list + (
        make_option(
            "-n",
            "--number",
            type="int",
            dest="sample_num",
            help="number of data samples consisting training and test sets",
        ),
        make_option(
            "-p",
            "--project",
            type="string",
            dest="project",
            help="project name",
        ),
    )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()

            # Get project of raise exception.
            project_name = options['project']
            if project_name is None:
                raise Exception('project not specified')

            # Load meme and user id's.
            memes = Meme.objects.filter(depth__gte=1)
            user_ids = UserAccount.objects.values_list('id', flat=True)

            if options['sample_num']:
                meme_ids = []
                # Repeat sampling if the sampled users do not have enough memes.
                while not meme_ids:
                    # Sample user id's and get their memes. Sample memes from this set.
                    self.stdout.write('sampling data ...')
                    sample_num = options['sample_num']
                    users_num = sample_num
                    user_samples = list(np.random.choice(user_ids, users_num, replace=False))
                    user_memes = memes.filter(postmeme__post__author__in=user_samples).distinct().values_list('id',
                                                                                                              flat=True)
                    try:
                        meme_ids = list(np.random.choice(user_memes, sample_num, replace=False))
                    except ValueError:
                        pass
            else:
                # Get all memes.
                self.stdout.write('sampling data ...')
                meme_ids = memes.values_list('id', flat=True)

            # Separate training and test sets.
            train_num = int(2.0 / 3 * len(meme_ids))
            train_set = meme_ids[:train_num]
            test_set = meme_ids[train_num:]

            project = Project(project_name)
            project.save_data(test_set, train_set)

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
