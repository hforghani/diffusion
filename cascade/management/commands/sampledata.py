# -*- coding: utf-8 -*-
import traceback
from django.core.management.base import BaseCommand
import time
import numpy as np
from cascade.models import Project
from crud.models import UserAccount, Meme


class Command(BaseCommand):
    help = 'Sample a subset of data and separate training and test sets and save them into the file'

    def add_arguments(self, parser):
        parser.add_argument(
            "-n",
            "--number",
            type=int,
            dest="sample_num",
            help="number of data samples consisting training and test sets",
        )
        parser.add_argument(
            "-r",
            "--ratio",
            type=float,
            dest="ratio",
            default=2.0 / 3,
            help="ratio of number of training set to number of all samples",
        )
        parser.add_argument(
            "-p",
            "--project",
            type=str,
            dest="project",
            help="project name",
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
                # TODO: shuffle randomly.
                self.stdout.write('sampling data ...')
                meme_ids = memes.values_list('id', flat=True)

            # Separate training and test sets.
            ratio = options['ratio']
            train_num = int(ratio * len(meme_ids))
            meme_ids = [int(m_id) for m_id in meme_ids]
            train_set = meme_ids[:train_num]
            test_set = meme_ids[train_num:]

            project = Project(project_name)
            project.save_data(test_set, train_set)

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
