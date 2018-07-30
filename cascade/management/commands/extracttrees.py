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
from cascade.models import CascadeTree
from crud.models import Reshare, UserAccount, Post, Meme
from scipy import sparse


class Command(BaseCommand):
    help = 'Extract cascade trees for all memes and save them into the file'

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()

            # Load trees from the json file.
            path = os.path.join(settings.BASEPATH, 'data', 'trees.json')
            trees = {}
            if os.path.exists(path):
                self.stdout.write('loading trees ...')
                trees = json.load(open(path, 'r'))
                trees = {int(key): value for key, value in trees.items()}

            # Check if all meme trees are in trees dictionary. Extract cascade tree for the ones not exist.
            self.stdout.write('checking if any tree not extracted ...')
            i = 0
            if len(trees.keys()) < Meme.objects.count():
                meme_ids = Meme.objects.exclude(id__in=trees.keys()).values_list('id', flat=True)
                self.stdout.write('extracting cascade trees for %d memes ...' % meme_ids.count())
                t0 = time.time()

                for meme_id in meme_ids:
                    i += 1
                    tree = CascadeTree().extract_cascade(meme_id).get_dict()
                    trees[meme_id] = tree
                    if i % 1000 == 0:
                        json.dump(trees, open(path, 'w'), indent=4)
                        self.stdout.write('%d memes done: %.2f m' % (i, (time.time() - t0) / 60))
                        t0 = time.time()

                json.dump(trees, open(path, 'w'), indent=4)

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
