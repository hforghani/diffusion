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
                #self.calc_depths(options['null'])
                self.calc_depths()

            self.stdout.write('command done in %.2f min' % ((time.time() - start) / 60.0))
        except:
            self.stdout.write(traceback.format_exc())
            raise

    def calc_depths(self):
        depths = {}
        tree_nodes = {}  # dictionary of meme id's to its tree nodes data. Each tree nodes data is a dictionary of user id's to its depth.

        reshares = mongodb.reshares.find({},
            {'_id': 0, 'post_id': 1, 'reshared_post_id': 1, 'user_id': 1, 'ref_user_id': 1}).sort(
            'datetime')

        count = reshares.count()
        self.stdout.write('number of all reshares: {}'.format(count))
        i = 0
        t0 = time.time()
        step = 10 ** 5
        save_step = 10 ** 6

        for resh in reshares:
            i += 1
            src_uid, dest_uid = resh['ref_user_id'], resh['user_id']

            if src_uid != dest_uid:
                ref_memes = {m['meme_id'] for m in
                             mongodb.postmemes.find({'post_id': resh['reshared_post_id']}, {'meme_id': 1})}
                memes = {m['meme_id'] for m in
                         mongodb.postmemes.find({'post_id': resh['post_id']}, {'meme_id': 1})}
                common_memes = ref_memes & memes

                for meme_id in common_memes:
                    if meme_id not in depths:
                        depths[meme_id] = 1
                        tree_nodes[meme_id] = {src_uid: 0, dest_uid: 1}
                        #self.stdout.write('meme {} has depth 1'.format(meme_id))

                    else:
                        nodes = tree_nodes[meme_id]
                        if src_uid not in nodes:
                            nodes[src_uid] = 0

                        if dest_uid not in nodes:
                            depth = nodes[src_uid] + 1
                            nodes[dest_uid] = depth
                            if depth > depths[meme_id]:
                                depths[meme_id] = depth
                                self.stdout.write('meme {} has now depth {}'.format(meme_id, depths[meme_id]))

            if i % step == 0:
                t = time.time() - t0
                avg = t / i
                rem = avg * (count - i)
                if rem > 60 * 48:
                    rem_str = '{:.0f} days'.format(rem / (60 * 24))
                elif rem > 60:
                    rem_str = '{:.0f} hours'.format(rem / 60)
                else:
                    rem_str = '{:.0f} minutes'.format(rem)
                self.stdout.write(
                    '{} reshares done. mean time: {:.0f} s per {}. estimated remaining time: {}'
                    .format(i, avg * step, step, rem_str))

            if i % save_step == 0:
                self.stdout.write('saving temp data ...')
                with open('data/i.json', 'w') as f:
                    json.dump({'i': i}, f)
                with open('data/depths.json', 'w') as f:
                    json.dump({str(key): value for key, value in depths.items()}, f, indent=4)
                with open('data/tree_nodes.json', 'w') as f:
                    json.dump({str(key): {str(user_id): value for user_id, value in nodes.items()} for key, nodes in
                               tree_nodes.items()}, f, indent=4)

        self.stdout.write('saving non-zero depths ...')
        for meme_id, depth in depths.items():
            mongodb.memes.find_one_and_update({'_id': meme_id}, {'$set': {'depth': depth}})

        self.stdout.write('saving zero depths ...')
        mongodb.memes.update_many({'depth': None}, {'$set': {'depth': 0}})


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
                    mongodb.memes.find_one_and_update({'_id': meme_id}, {'$set': {'depth': tree.depth}})
                    i += 1
                    if i % 100 == 0:
                        self.stdout.write('%d memes done' % i)
                    json_str = '{'
