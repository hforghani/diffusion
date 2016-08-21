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
    )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()

            self.stdout.write('sampling data ...')
            memes = Meme.objects.filter(depth__gte=1)

            if options['sample_num']:
                sample_num = options['sample_num']
            else:
                sample_num = memes.count()

            samples = list(np.random.choice(memes.values_list('id', flat=True), sample_num, replace=False))
            train_num = 2.0 / 3 * memes.count()
            train_set = samples[:train_num]
            test_set = samples[train_num:]

            data = {'training': train_set, 'test': test_set}
            sample_path = os.path.join(settings.BASEPATH, 'data', 'samples.json')
            json.dump(data, open(sample_path, 'w'), indent=4)

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
