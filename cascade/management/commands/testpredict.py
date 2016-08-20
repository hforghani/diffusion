# -*- coding: utf-8 -*-
import json
from optparse import make_option
import os
import traceback
from django.conf import settings
from django.core.management.base import BaseCommand
import time
import numpy as np
from cascade.validation import Validation
from cascade.saito import Saito
from crud.models import Meme, UserAccount
from cascade.models import CascadeTree


class Command(BaseCommand):
    help = 'Test information diffusion prediction'

    option_list = BaseCommand.option_list + (
        make_option('-t',
                    '--train',
                    action='store_true',
                    dest='train',
                    default=False,
                    help='Test on training set instead of test set'),
    )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()

            train_set_path = os.path.join(settings.BASEPATH, 'data', 'train_set.json')
            train_set = json.load(open(train_set_path, 'r'))
            if options['train']:
                test_set = Meme.objects.filter(id__in=train_set)
            else:
                test_set = Meme.objects.filter(count__gt=500).exclude(id__in=train_set)
            test_set = test_set.filter(depth__gt=0).order_by('id')
            self.stdout.write('test set size = %d' % test_set.count())

            # Load trees from the json file.
            path = os.path.join(settings.BASEPATH, 'data', 'trees.json')
            trees = {}
            if os.path.exists(path):
                self.stdout.write('loading trees ...')
                trees = json.load(open(path, 'r'))
                trees = {long(key): value for key, value in trees.items()}

            # Check if all meme trees are in trees dictionary. Extract the cascade trees which are not in trees.
            self.stdout.write('checking trees includes test set ...')
            i = 0
            if not set(test_set.values_list('id', flat=True)) <= set(trees.keys()):
                self.stdout.write('extracting cascade trees ...')
                for meme in test_set.exclude(id__in=trees.keys()):
                    i += 1
                    t0 = time.time()
                    tree = CascadeTree().extract_cascade(meme).get_dict()
                    self.stdout.write('meme %d done: %.2f s' % (i, time.time() - t0))
                    trees[meme.id] = tree
                    if i % 100 == 0:
                        json.dump(trees, open(path, 'w'), indent=4)
                json.dump(trees, open(path, 'w'), indent=4)

            # Convert tree dictionaries to tree objects.
            self.stdout.write('converting trees to objects ...')
            trees = {meme_id: CascadeTree().from_dict(tree) for meme_id, tree in trees.items()}

            i = 0
            prp1_list = []
            prp2_list = []

            # Test the prediction on the test set.
            for meme in test_set:
                tree = trees[meme.id]

                # Copy roots in a new tree.
                initial_tree = tree.copy()
                for node in initial_tree.tree:
                    node.children = []

                # Predict remaining nodes.
                user_ids = UserAccount.objects.values_list('id', flat=True).order_by('id')
                model = Saito().fit(initial_tree)
                res_tree = model.predict(user_ids)

                # Evaluate the result.
                res_nodes = set(res_tree.node_ids())
                true_nodes = set(tree.node_ids())
                initial_nodes = set(initial_tree.node_ids())
                res_output = res_nodes - initial_nodes
                true_output = true_nodes - initial_nodes
                meas = Validation(res_output, true_output)
                prec = meas.precision()
                rec = meas.recall()
                prp = meas.prp(model.weight_sum)
                prp1 = prp[0] if prp else 0
                prp2 = prp[1] if len(prp) > 1 else 0
                prp1_list.append(prp1)
                prp2_list.append(prp2)

                i += 1
                self.stdout.write(
                    'meme %d: %d outputs, %d true, precision = %.3f, recall = %.3f, prp = (%.3f, %.3f, ...)' % (
                        i, len(res_output), len(true_output), prec, rec, prp1, prp2))

            self.stdout.write('prp1 avg = %.3f' % np.mean(np.array(prp1_list)))
            self.stdout.write('prp2 avg = %.3f' % np.mean(np.array(prp2_list)))

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
