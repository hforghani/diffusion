# -*- coding: utf-8 -*-
from optparse import make_option
import traceback
from django.core.management.base import BaseCommand
import time
import numpy as np
from cascade.validation import Validation
from cascade.saito import Saito
from cascade.models import Project, AsLT
from mln.models import MLN


class Command(BaseCommand):
    help = 'Test information diffusion prediction'

    option_list = BaseCommand.option_list + (
        make_option(
            "-p",
            "--project",
            type="string",
            dest="project",
            help="project name",
        ),
        make_option(
            "-m",
            "--method",
            type="string",
            dest="method",
            help="the method by which we want to test. values: saito, mln",
        ),
    )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()

            # Get project or raise exception.
            project_name = options['project']
            if project_name is None:
                raise Exception('project not specified')
            project = Project(project_name)

            # Get the method or raise exception.
            method = options['method']
            if method is None:
                raise Exception('method not specified')

            # Load training and test sets and cascade trees.
            train_set, test_set = project.load_data()
            trees = project.load_trees()
            self.stdout.write('test set size = %d' % len(test_set))

            predicted_trees = {}
            if method == 'mln':
                self.stdout.write('loading mln results ...')
                file_path = 'D:/University Stuff/social/code/pracmln/myprojects/social/results/small-3memes-gibbs.results'
                predicted_trees = MLN(project, threshold=30).load_results(file_path, trees)
                test_set = set(test_set) & set(predicted_trees.keys())

            i = 0
            prp1_list = []
            prp2_list = []
            all_res_nodes = []
            all_true_nodes = []

            # Test the prediction on the test set.
            for meme_id in test_set:
                tree = trees[meme_id]

                # Copy roots in a new tree.
                initial_tree = tree.copy()
                for node in initial_tree.tree:
                    node.children = []

                # Predict remaining nodes.
                if method == 'mln':
                    res_tree = predicted_trees[meme_id]
                elif method == 'saito':
                    model = Saito(project).fit(initial_tree)
                    res_tree = model.predict()
                elif method == 'avg':
                    model = AsLT(project).fit(initial_tree)
                    res_tree = model.predict()
                else:
                    raise Exception('invalid method "%s"' % method)

                # Get predicted and true nodes.
                res_nodes = set(res_tree.node_ids())
                true_nodes = set(tree.node_ids())
                initial_nodes = set(initial_tree.node_ids())
                res_output = res_nodes - initial_nodes
                true_output = true_nodes - initial_nodes

                # Put meme id str at the beginning of user id's to make in unique.
                all_res_nodes.extend({str(meme_id) + str(node) for node in res_output})
                all_true_nodes.extend({str(meme_id) + str(node) for node in true_output})

                # Evaluate the result.
                meas = Validation(res_output, true_output)
                prec = meas.precision()
                rec = meas.recall()

                if method != 'mln':
                    prp = meas.prp(model.weight_sum)
                    prp1 = prp[0] if prp else 0
                    prp2 = prp[1] if len(prp) > 1 else 0
                    prp1_list.append(prp1)
                    prp2_list.append(prp2)

                i += 1
                log = 'meme %d: %d outputs, %d true, precision = %.3f, recall = %.3f' % (
                    i, len(res_output), len(true_output), prec, rec)
                if method != 'mln':
                    log += ', prp = (%.3f, %.3f, ...)' % (prp1, prp2)
                self.stdout.write(log)

            # Evaluate total results.
            meas = Validation(all_res_nodes, all_true_nodes)
            prec, rec, f1 = meas.precision(), meas.recall(), meas.f1()
            self.stdout.write('total: %d outputs, %d true, precision = %.3f, recall = %.3f, f1 = %.3f' % (
                len(all_res_nodes), len(all_true_nodes), prec, rec, f1))

            if method != 'mln':
                self.stdout.write('prp1 avg = %.3f' % np.mean(np.array(prp1_list)))
                self.stdout.write('prp2 avg = %.3f' % np.mean(np.array(prp2_list)))

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
