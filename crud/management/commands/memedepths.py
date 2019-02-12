# -*- coding: utf-8 -*-
import json
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
from crud.mongo import mongodb


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
        query = {'depth': None} if just_null else {}
        memes = mongodb.memes.find(query, {'_id': 1})

        count = memes.count()
        self.stdout.write('number of memes to calculate depths = %d' % count)
        i = 0
        step = 1000
        t0 = time.time()
        times = []

        for meme in memes:
            tree = CascadeTree().extract_cascade(meme['_id'])
            mongodb.memes.find_one_and_update({'_id': meme['_id']}, {'$set': {'depth': tree.depth}})

            i += 1
            if i % step == 0:
                t = time.time() - t0
                if len(times) >= 10:
                    times.pop(0)
                times.append(t)

                avg = sum(times) / len(times)
                rem = (count - i) * avg / (60 * step)

                if rem > 60 * 48:
                    rem_str = '{:.0f} days'.format(rem / (60 * 24))
                elif rem > 60:
                    rem_str = '{:.0f} hours'.format(rem / 60)
                else:
                    rem_str = '{:.0f} minutes'.format(rem)

                self.stdout.write(
                    '{} memes done. mean time: {:.0f} s per {}. estimated remaining time: {}'
                    .format(i, avg, step, rem_str))
                t0 = time.time()


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
