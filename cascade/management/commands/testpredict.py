# -*- coding: utf-8 -*-
import json
from optparse import make_option
import os
import traceback
from django.conf import settings
from django.core.management.base import BaseCommand
import time
from matplotlib import pyplot
import numpy as np
from cascade.saito import Saito
from crud.models import Meme
from cascade.models import CascadeTree, AsLT


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

            train_set_path = os.path.join(settings.BASEPATH, 'resources', 'samples.json')
            train_set = json.load(open(train_set_path, 'r'))
            if options['train']:
                test_set = Meme.objects.filter(id__in=train_set).order_by('id')
            else:
                test_set = Meme.objects.filter(count__gt=500).exclude(id__in=train_set).order_by('id')
            self.stdout.write('test set size = %d' % test_set.count())

            # Load trees from the json file.
            path = os.path.join(settings.BASEPATH, 'resources', 'trees.json')
            trees = {}
            if os.path.exists(path):
                self.stdout.write('loading trees ...')
                trees = json.load(open(path, 'r'))
                trees = {long(key): value for key, value in trees.items()}

            # Check if all meme trees are in trees dictionary. Extract the cascade trees which are not in trees.
            self.stdout.write('checking trees includes test set ...')
            i = 0
            if not set(test_set.values_list('id', flat=True)) == trees.keys():
                for meme in test_set:
                    if meme.id not in trees:
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

            precisions = []
            recalls = []
            i = 0

            # Test the prediction on the test set.
            for meme in test_set:
                t0 = time.time()
                tree = trees[meme.id]
                print tree

                # Copy roots in a new tree.
                initial_tree = tree.copy()
                for node in initial_tree.tree:
                    node.children = []
                self.stdout.write('copy time: %.2f s' % (time.time() - t0))

                # Predict remaining nodes.
                t0 = time.time()
                res_tree = Saito().fit(initial_tree).predict()

                # Evaluate the result.
                res_nodes = set(res_tree.node_ids())
                true_nodes = set(tree.node_ids())
                initial_nodes = set(initial_tree.node_ids())
                res_output = res_nodes - initial_nodes
                true_output = true_nodes - initial_nodes
                tp = res_output.intersection(true_output)
                if not res_output:
                    precision = 1
                else:
                    precision = float(len(tp)) / len(res_output)
                precisions.append(precision)
                if not true_output:
                    recall = 1
                else:
                    recall = float(len(tp)) / len(true_output)
                recalls.append(recall)

                i += 1
                self.stdout.write('prediction time: %.2f s' % (time.time() - t0))
                self.stdout.write(
                    'meme %d: %d outputs, %d results, precision = %f, recall = %f' % (
                        i, len(true_output), len(res_output), precision, recall))

            # Plot the results.
            pyplot.scatter(recalls, precisions)
            pyplot.axis([0, 1, 0, 1])
            pyplot.xlabel('recall')
            pyplot.ylabel('precision')
            pyplot.show()

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
