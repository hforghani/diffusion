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
    help = 'Sample training set and save it into the file'

    option_list = BaseCommand.option_list + (
        make_option(
            "-n",
            "--number",
            type="int",
            dest="train_num",
            help="number of training samples",
        ),
    )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()

            train_set_path = os.path.join(settings.BASEPATH, 'data', 'train_set.json')
            self.stdout.write('sampling training set ...')
            if options['train_num']:
                train_count = options['train_num']
            else:
                train_count = 2.0 / 3 * Meme.objects.filter(depth__gte=1).count()
            train_set = list(
                np.random.choice(Meme.objects.filter(depth__gte=1).values_list('id', flat=True), train_count,
                                 replace=False))
            json.dump(train_set, open(train_set_path, 'w'), indent=4)

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
