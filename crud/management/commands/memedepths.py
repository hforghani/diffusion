# -*- coding: utf-8 -*-
import json
import math
import multiprocessing
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
from crud.models import Meme


class Command(BaseCommand):
    help = 'Calculate meme depths.'

    def add_arguments(self, parser):
        parser.add_argument(
            '-n',
            '--null',
            action='store_true',
            default=False,
            help='Calculate depths of o',
        )

    def handle(self, *args, **options):
        try:
            start = time.time()

            trees_path = os.path.join(settings.BASEPATH, 'data', 'trees.json')
            if os.path.exists(trees_path):
                self.set_depths_by_trees_data(trees_path)
            else:
                self.stdout.write('NOTICE: Trees data not found. We calculate depths from scratch. It may take too '
                                  'much time. You can also stop this command and execute "exctracttrees" command '
                                  'and then this command.')
                self.calc_depths(options['null'])

            self.stdout.write('command done in %.2f min' % ((time.time() - start) / 60.0))
        except:
            self.stdout.write(traceback.format_exc())
            raise

    def calc_depths(self, just_null=False):
        i = 0
        t0 = time.time()
        memes = Meme.objects
        if just_null:
            memes = memes.filter(depth__isnull=True)
        self.stdout.write('number of memes to calculate depths = %d' % memes.count())
        for meme in memes.iterator():
            tree = CascadeTree().extract_cascade(meme.id)
            meme.depth = tree.depth
            meme.save()
            i += 1
            if i % 100 == 0:
                self.stdout.write('%d memes done. mean time: %.2f s' % (i, (time.time() - t0) / i * 100))

    def set_depths_by_trees_data(self, trees_path):
        self.stdout.write('loading trees ...')
        with open(trees_path, 'r') as f:
            i = 0
            json_str = '{'
            for line in f:
                if line in ['{\n', '}\n', '}']:
                    continue
                line = line.strip()
                if line != '],':
                    json_str += line
                else:
                    json_str += ']}'
                    data = json.loads(json_str)
                    meme_id = int(list(data.keys())[0])
                    tree = CascadeTree().from_dict(list(data.values())[0])
                    Meme.objects.filter(id=meme_id).update(depth=tree.depth)
                    i += 1
                    if i % 100 == 0:
                        self.stdout.write('%d memes done' % i)
                    json_str = '{'
