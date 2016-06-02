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
from crud.models import Meme
from diffusion.models import CascadeTree, CascadePredictor


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

            train_set_path = os.path.join(settings.BASEPATH, 'resources', 'diff_samples.json')
            train_set = json.load(open(train_set_path, 'r'))
            if options['train']:
                test_set = Meme.objects.filter(id__in=train_set)
            else:
                test_set = Meme.objects.filter(count__gt=500).exclude(id__in=train_set).order_by('id')
            self.stdout.write('test set size = %d' % test_set.count())

            precisions = []
            recalls = []
            i = 0

            for meme in test_set:
                tree = CascadeTree().extract_cascade(meme)

                # Copy roots in a new tree.
                initial_tree = tree.copy()
                for node in initial_tree.tree['children']:
                    node['children'] = []

                # Predict remaining nodes.
                res_tree = CascadePredictor(initial_tree).predict()

                # Evaluate result.
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
                self.stdout.write(
                    'meme %d: %d output nodes, precision = %f, recall = %f' % (i, len(true_output), precision, recall))

            pyplot.scatter(recalls, precisions)
            pyplot.axis([0, 1, 0, 1])
            pyplot.xlabel('recall')
            pyplot.ylabel('precision')
            pyplot.show()

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
