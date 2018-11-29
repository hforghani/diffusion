# -*- coding: utf-8 -*-
import json
from multiprocessing.pool import Pool
import os
import traceback
import django
from django.apps import apps
from django.conf import settings
from django.core.management.base import BaseCommand
import time
import math

if not apps.ready and not settings.configured:
    django.setup()

from cascade.models import CascadeTree
from crud.models import Meme


def extract_cascades(meme_ids, process_num):
    t0 = time.time()
    trees = {}
    i = 0

    for meme_id in meme_ids:
        i += 1
        tree = CascadeTree().extract_cascade(meme_id).get_dict()
        trees[meme_id] = tree
        if i % 1000 == 0:
            print('process {}: {} memes done: {:.0f} s'.format(process_num, i, time.time() - t0))
            t0 = time.time()

    return trees


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

            # Check if all meme trees are in trees dictionary. Extract cascade tree for the ones which not exist.
            self.stdout.write('checking if any tree not extracted ...')
            if len(trees.keys()) < Meme.objects.count():
                meme_ids = Meme.objects.values_list('id', flat=True)
                meme_ids = list(set(meme_ids) - set(trees.keys()))
                self.stdout.write('extracting cascade trees for %d memes ...' % len(meme_ids))

                # Distribute the extraction into some simultaneous processes.
                process_count = 2
                step = int(math.ceil(len(meme_ids) / process_count))
                pool = Pool(processes=process_count)
                results = []

                for process_num in range(1, step + 1):
                    cur_meme_ids = meme_ids[(process_num - 1) * step: process_num * step]
                    res = pool.apply_async(extract_cascades, (cur_meme_ids, process_num))
                    results.append(res)

                pool.close()
                pool.join()

                # Gather and add up the results of the processes.
                for res in results:
                    cur_trees = res.get()
                    trees.update(cur_trees)

                # Save the results into the file.
                json.dump(trees, open(path, 'w'), indent=1)

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
