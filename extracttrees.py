# -*- coding: utf-8 -*-
import json
import sys
import os
import traceback
import django
from django.apps import apps
from django.conf import settings
from django.core.management.base import BaseCommand
import time

if not apps.ready and not settings.configured:
    django.setup()

from cascade.models import CascadeTree


def extract_cascades(meme_ids, process_num, stdout=sys.stdout):
    try:
        t0 = time.time()
        trees = {}
        i = 0

        for meme_id in meme_ids:
            i += 1
            tree = CascadeTree().extract_cascade(meme_id).get_dict()
            trees[meme_id] = tree
            if i % 100 == 0:
                stdout.write('process {}: {} memes done: {:.0f} s'.format(process_num, i, time.time() - t0))
                t0 = time.time()

        return trees
    except:
        traceback.print_exc()


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
                if trees is not None:
                    trees = {int(key): value for key, value in trees.items()}
                else:
                    trees = {}

            # Check if all meme trees are in trees dictionary. Extract cascade tree for the ones which not exist.
            self.stdout.write('checking if any tree not extracted ...')
            if len(trees.keys()) < Meme.objects.count():
                meme_ids = Meme.objects.values_list('id', flat=True)
                meme_ids = list(set(meme_ids) - set(trees.keys()))
                self.stdout.write('extracting cascade trees for %d memes ...' % len(meme_ids))

                trees = extract_cascades(meme_ids, 1, self.stdout)

                # Save the results into the file.
                self.stdout.write('writing to file ...')
                json.dump(trees, open(path, 'w'), indent=1)

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
