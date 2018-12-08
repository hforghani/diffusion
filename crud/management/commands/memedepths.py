# -*- coding: utf-8 -*-
import json
import os
import traceback

from django.conf import settings
from django.core.management.base import BaseCommand
import time
from cascade.models import CascadeTree
from crud.models import Meme


class Command(BaseCommand):
    help = 'Calculate meme depths.'

    def handle(self, *args, **options):
        try:
            start = time.time()

            trees_path = os.path.join(settings.BASEPATH, 'data', 'trees.json')
            if os.path.exists(trees_path):
                self.set_depths_by_trees_data(trees_path)
            else:
                self.stdout.write('NOTICE: Trees data not found. We calculate depths from scratch. It may tak too '
                                  'much time. You can also stop this command and execute "ectracttrees" command '
                                  'and then this command.')
                self.calc_depths()

            self.stdout.write('command done in %.2f min' % ((time.time() - start) / 60.0))
        except:
            self.stdout.write(traceback.format_exc())
            raise

    def calc_depths(self):
        self.stdout.write('meme count = %d' % Meme.objects.count())
        i = 0
        # for meme in Meme.objects.filter(depth__isnuall=True).iterator():
        for meme in Meme.objects.iterator():
            tree = CascadeTree().extract_cascade(meme.id)
            meme.depth = tree.depth
            meme.save()
            i += 1
            if i % 100 == 0:
                self.stdout.write('%d memes done' % i)

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
                    # if meme_id in [229223, 4965313, 319945]:
                    tree = CascadeTree().from_dict(list(data.values())[0])
                    # self.stdout.write('depth = %d' % tree.depth)
                    Meme.objects.filter(id=meme_id).update(depth=tree.depth)
                    i += 1
                    if i % 100 == 0:
                        self.stdout.write('%d memes done' % i)
                    json_str = '{'
