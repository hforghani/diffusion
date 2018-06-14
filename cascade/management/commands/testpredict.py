# -*- coding: utf-8 -*-
import logging
from optparse import make_option
import traceback
from django.conf import settings
from django.core.management.base import BaseCommand
import time
import numpy as np
from cascade.validation import Validation
from cascade.saito import Saito
from cascade.models import Project, AsLT
from memm.models import MEMMModel
from mln.models import MLN

logger = logging.getLogger('cascade.management.commands.testpredict')


class Command(BaseCommand):
    help = 'Test information diffusion prediction'

    option_list = BaseCommand.option_list + (
        #make_option(
        #    "-p",
        #    "--project",
        #    type="string",
        #    dest="project",
        #    help="project name",
        #),
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
            #projects = ['small0', 'small1', 'small3', 'small4', 'small5', 'small6', 'small7']
            projects = ['big']
            prec = []
            recall = []
            f1 = []

            for project_name in projects:
                ## Get project or raise exception.
                #project_name = options['project']
                #if project_name is None:
                #    raise Exception('project not specified')
                project = Project(project_name)

                # Get the method or raise exception.
                method = options['method']
                if method is None:
                    raise Exception('method not specified')

                measure = self.test(project, method)
                prec.append(measure.precision())
                recall.append(measure.recall())
                f1.append(measure.f1())

            if len(projects) > 1:
                logger.info('final precision = %.3f', np.mean(np.array(prec)))
                logger.info('final recall = %.3f', np.mean(np.array(recall)))
                logger.info('final f1 = %.3f', np.mean(np.array(f1)))
            logger.info('command done in %.2f min' % ((time.time() - start) / 60))
        except:
            logger.info(traceback.format_exc())
            raise

    def test(self, project, method):
        # Load training and test sets and cascade trees.
        train_set, test_set = project.load_train_test()
        trees = project.load_trees()
        logger.info('test set size = %d' % len(test_set))

        # Create and train the model if needed.
        if method == 'mln':
            logger.info('loading mln results ...')
            file_path = 'D:/University Stuff/social/code/pracmln/experiments/social/results/%s-gibbs.results' % project.project_name
            model = MLN(project)
            model.load_results(file_path)
            test_set = set(test_set) & set(model.edges.keys())
            threshold = settings.MLN_THRES
        elif method == 'memm':
            model = MEMMModel(project).fit(train_set)
            threshold = settings.MEMM_THRES
        elif method == 'saito':
            model = Saito(project)
            threshold = settings.ASLT_THRES
        elif method == 'avg':
            model = AsLT(project)
            threshold = settings.LT_THRES
        else:
            raise Exception('invalid method "%s"' % method)

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
            for node in initial_tree.roots:
                node.children = []

            # Predict remaining nodes.
            if method == 'mln':
                res_tree = model.predict(meme_id, initial_tree, threshold=threshold)
            else:
                res_tree = model.predict(initial_tree, threshold=threshold)

            # Evaluate the results.
            meas, prec, rec, res_output, true_output = self.evaluate(initial_tree, res_tree, tree)

            if method in ['saito', 'avg']:
                prp = meas.prp(model.probabilities)
                prp1 = prp[0] if prp else 0
                prp2 = prp[1] if len(prp) > 1 else 0
                prp1_list.append(prp1)
                prp2_list.append(prp2)

            # Put meme id str at the beginning of user id to make it unique.
            all_res_nodes.extend({'{}-{}'.format(meme_id, node) for node in res_output})
            all_true_nodes.extend({'{}-{}'.format(meme_id, node) for node in true_output})

            i += 1
            log = 'meme %d: %d outputs, %d true, precision = %.3f, recall = %.3f' % (
                i, len(res_output), len(true_output), prec, rec)
            if method in ['saito', 'avg']:
                log += ', prp = (%.3f, %.3f, ...)' % (prp1, prp2)
            logger.info(log)

        # Evaluate total results.
        meas = Validation(all_res_nodes, all_true_nodes)
        prec, rec, f1 = meas.precision(), meas.recall(), meas.f1()
        logger.info('total: %d outputs, %d true, precision = %.3f, recall = %.3f, f1 = %.3f' % (
            len(all_res_nodes), len(all_true_nodes), prec, rec, f1))

        if method in ['saito', 'avg']:
            logger.info('prp1 avg = %.3f' % np.mean(np.array(prp1_list)))
            logger.info('prp2 avg = %.3f' % np.mean(np.array(prp2_list)))

        return meas

    def evaluate(self, initial_tree, res_tree, tree):
        # Get predicted and true nodes.
        res_nodes = set(res_tree.node_ids())
        true_nodes = set(tree.node_ids())
        initial_nodes = set(initial_tree.node_ids())
        res_output = res_nodes - initial_nodes
        true_output = true_nodes - initial_nodes
        # Evaluate the result.
        meas = Validation(res_output, true_output)
        prec = meas.precision()
        rec = meas.recall()
        return meas, prec, rec, res_output, true_output
