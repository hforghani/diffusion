# -*- coding: utf-8 -*-
import logging
from multiprocessing import Pool
import traceback
import django
from django.apps import apps
from django.conf import settings
from django.core.management.base import BaseCommand
import time
import math
import numpy as np

if not apps.ready and not settings.configured:
    django.setup()

from cascade.avg import LTAvg
from cascade.validation import Validation
from cascade.saito import Saito
from cascade.models import Project
from memm.models import MEMMModel
from mln.models import MLN

logger = logging.getLogger('cascade.management.commands.testpredict')


def evaluate(initial_tree, res_tree, tree):
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


def test_meme(meme_ids, method, model, threshold, trees):
    try:
        prp1_list = []
        prp2_list = []
        all_res_nodes = []
        all_true_nodes = []

        for meme_id in meme_ids:
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
            meas, prec, rec, res_output, true_output = evaluate(initial_tree, res_tree, tree)

            if method in ['saito', 'avg']:
                prp = meas.prp(model.probabilities)
                prp1 = prp[0] if prp else 0
                prp2 = prp[1] if len(prp) > 1 else 0
                prp1_list.append(prp1)
                prp2_list.append(prp2)

            # Put meme id str at the beginning of user id to make it unique.
            all_res_nodes.extend({'{}-{}'.format(meme_id, node) for node in res_output})
            all_true_nodes.extend({'{}-{}'.format(meme_id, node) for node in true_output})

            log = 'meme %d: %d outputs, %d true, precision = %.3f, recall = %.3f' % (
                meme_id, len(res_output), len(true_output), prec, rec)
            if method in ['saito', 'avg']:
                log += ', prp = (%.3f, %.3f, ...)' % (prp1, prp2)
            print(log)

        return all_res_nodes, all_true_nodes, prp1_list, prp2_list

    except:
        print(traceback.format_exc())
        raise


class Command(BaseCommand):
    help = 'Test information diffusion prediction'

    def add_arguments(self, parser):
        # Named (optional) arguments
        #parser.add_argument(
        #    "-p",
        #    "--project",
        #    type="string",
        #    dest="project",
        #    help="project name",
        #)
        parser.add_argument(
            "-m",
            "--method",
            type=str,
            dest="method",
            help="the method by which we want to test. values: saito, mln",
        )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        start = time.time()
        projects = ['small0', 'small1', 'small3', 'small4', 'small5', 'small6', 'small7']
        #projects = ['big']
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

    def test(self, project, method):
        # Load training and test sets and cascade trees.
        train_set, test_set = project.load_train_test()
        trees = project.load_trees()
        logger.info('test set size = %d' % len(test_set))

        # Create and train the model if needed.
        if method == 'mln':
            logger.info('loading mln results ...')
            model = MLN(project)
            model.load_results()
            threshold = settings.MLN_THRES
        elif method == 'memm':
            model = MEMMModel(project).fit(train_set)
            threshold = settings.MEMM_THRES
        elif method == 'saito':
            model = Saito(project)
            threshold = settings.ASLT_THRES
        elif method == 'avg':
            model = LTAvg(project)
            threshold = settings.LTAVG_THRES
        else:
            raise Exception('invalid method "%s"' % method)

        ## Create a process pool to distribute the prediction.
        process_count = 3
        pool = Pool(processes=process_count)
        step = int(math.ceil(float(len(test_set)) / process_count))
        results = []
        for j in range(0, len(test_set), step):
            meme_ids = test_set[j: j + step]
            res = pool.apply_async(test_meme, (meme_ids, method, model, threshold, trees))
            results.append(res)

        pool.close()
        pool.join()

        prp1_list = []
        prp2_list = []
        all_res_nodes = []
        all_true_nodes = []

        # Collect results of the processes.
        for res in results:
            r1, r2, r3, r4 = res.get()
            all_res_nodes.extend(r1)
            all_true_nodes.extend(r2)
            prp1_list.extend(r3)
            prp2_list.extend(r4)

        #all_res_nodes, all_true_nodes, prp1_list, prp2_list = test_meme(test_set, method, model, threshold, trees)

        # Evaluate total results.
        meas = Validation(all_res_nodes, all_true_nodes)
        prec, rec, f1 = meas.precision(), meas.recall(), meas.f1()
        logger.info('project %s: %d outputs, %d true, precision = %.3f, recall = %.3f, f1 = %.3f' % (
            project.project_name, len(all_res_nodes), len(all_true_nodes), prec, rec, f1))

        if method in ['saito', 'avg']:
            logger.info('prp1 avg = %.3f' % np.mean(np.array(prp1_list)))
            logger.info('prp2 avg = %.3f' % np.mean(np.array(prp2_list)))

        return meas
