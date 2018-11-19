# -*- coding: utf-8 -*-
import logging
from multiprocessing import Pool
import traceback
import django
from django.apps import apps
from django.conf import settings
from django.core.management import CommandError
from django.core.management.base import BaseCommand
import time
import math
import numpy as np
from matplotlib import pyplot

from mln.file_generators import FileCreator

if not apps.ready and not settings.configured:
    django.setup()

from cascade.avg import LTAvg
from cascade.validation import Validation
from cascade.saito import Saito
from cascade.models import Project
from memm.models import MEMMModel
from mln.models import MLN

logger = logging.getLogger('cascade.management.commands.testpredict')


def evaluate(initial_tree, res_tree, tree, all_nodes, verbosity=settings.VERBOSITY):
    # Get predicted and true nodes.
    res_nodes = set(res_tree.node_ids())
    true_nodes = set(tree.node_ids())
    initial_nodes = set(initial_tree.node_ids())
    res_output = res_nodes - initial_nodes
    true_output = true_nodes - initial_nodes

    # Evaluate the result.
    meas = Validation(res_output, true_output, all_nodes)
    prec = meas.precision()
    rec = meas.recall()
    return meas, prec, rec, res_output, true_output


def test_meme(meme_ids, method, model, threshold, trees, all_node_ids, verbosity=settings.VERBOSITY):
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
            if method in ['mlnprac', 'mlnalch']:
                res_tree = model.predict(meme_id, initial_tree, threshold=threshold, verbosity=verbosity)
            else:
                res_tree = model.predict(initial_tree, threshold=threshold)

            # Evaluate the results.
            meas, prec, rec, res_output, true_output = evaluate(initial_tree, res_tree, tree, all_node_ids)

            if method in ['saito', 'avg']:
                prp = meas.prp(model.probabilities)
                prp1 = prp[0] if prp else 0
                prp2 = prp[1] if len(prp) > 1 else 0
                prp1_list.append(prp1)
                prp2_list.append(prp2)

            # Put meme id str at the beginning of user id to make it unique.
            all_res_nodes.extend({'{}-{}'.format(meme_id, node) for node in res_output})
            all_true_nodes.extend({'{}-{}'.format(meme_id, node) for node in true_output})

            if verbosity > 2:
                log = 'meme %d: %d outputs, %d true, precision = %.3f, recall = %.3f' % (
                    meme_id, len(res_output), len(true_output), prec, rec)
                if method in ['saito', 'avg']:
                    log += ', prp = (%.3f, %.3f, ...)' % (prp1, prp2)
                logger.info(log)
                # logger.info('output: %s', res_output)
                # logger.info('true: %s', true_output)

        return all_res_nodes, all_true_nodes, prp1_list, prp2_list

    except:
        print(traceback.format_exc())
        raise


class Command(BaseCommand):
    help = 'Test information diffusion prediction'

    def add_arguments(self, parser):
        parser.add_argument(
            "-p",
            "--project",
            type=str,
            dest="project",
            help="project name or multiple comma-separated project names",
        )
        parser.add_argument(
            "-m",
            "--method",
            type=str,
            dest="method",
            help="the method by which we want to test. values: saito, avg, mlnalch, mlnprac, memm",
        )
        parser.add_argument(
            "-a",
            "--all",
            action='store_true',
            dest="all_thresholds",
            default=False,
            help="test the method for all threshold values and show the charts",
        )

    thresholds = {
        'mlnprac': settings.MLNPRAC_THRES,
        'mlnalch': settings.MLNALCH_THRES,
        'memm': settings.MEMM_THRES,
        'saito': settings.ASLT_THRES,
        'avg': settings.LTAVG_THRES
    }

    def __init__(self):
        super(Command, self).__init__()
        self.verbosity = settings.VERBOSITY

    def handle(self, *args, **options):
        start = time.time()
        projects = options['project'].split(',')
        self.verbosity = options['verbosity'] if options['verbosity'] is not None else settings.VERBOSITY

        # Get the method or raise exception.
        method = options['method']
        if method is None:
            raise CommandError('method not specified')

        try:
            settings_thr = self.thresholds[method]
        except KeyError:
            raise CommandError('invalid method "%s"' % method)

        if options['all_thresholds']:
            count = 40
            step = settings_thr / count * 2
            thresholds = [step * i for i in range(count)]
        else:
            thresholds = [settings_thr]

        final_prec = []
        final_recall = []
        final_f1 = []
        final_fpr = []

        for thr in thresholds:
            if options['all_thresholds']:
                if self.verbosity:
                    logger.info('{0} THRESHOLD = {1} {0}'.format('=' * 20, thr))
            prec = []
            recall = []
            f1 = []
            fpr = []

            for project_name in projects:
                project = Project(project_name)
                measure = self.test(project, method, thr)
                prec.append(measure.precision())
                recall.append(measure.recall())
                f1.append(measure.f1())
                fpr.append(measure.fpr())

            fprec = np.mean(np.array(prec))
            frecall = np.mean(np.array(recall))
            ff1 = np.mean(np.array(f1))
            ffpr = np.mean(np.array(fpr))
            final_prec.append(fprec)
            final_recall.append(frecall)
            final_f1.append(ff1)
            final_fpr.append(ffpr)

            if len(projects) > 1 and self.verbosity:
                logger.info('final precision = %.3f', fprec)
                logger.info('final recall = %.3f', frecall)
                logger.info('final f1 = %.3f', ff1)
                logger.info('final fpr = %.3f', ffpr)

        if options['all_thresholds']:
            pyplot.figure(1)
            pyplot.subplot(221)
            pyplot.plot(thresholds, final_prec)
            pyplot.title('precision')
            pyplot.subplot(222)
            pyplot.plot(thresholds, final_recall)
            pyplot.title('recall')
            pyplot.subplot(223)
            pyplot.plot(thresholds, final_f1)
            pyplot.title('F1')
            pyplot.subplot(224)
            pyplot.plot(final_fpr, final_recall)
            pyplot.title('ROC curve')
            pyplot.show()

        if self.verbosity:
            logger.info('command done in %.2f min' % ((time.time() - start) / 60))

    def test(self, project, method, threshold):
        # Load training and test sets and cascade trees.
        train_set, test_set = project.load_train_test()
        trees = project.load_trees()
        all_node_ids = project.get_all_nodes()

        if self.verbosity > 1:
            logger.info('test set size = %d' % len(test_set))

        # Create and train the model if needed.
        if method == 'mlnprac':
            model = MLN(project, format=FileCreator.FORMAT_PRACMLN)
        elif method == 'mlnalch':
            model = MLN(project, format=FileCreator.FORMAT_ALCHEMY2)
        elif method == 'memm':
            model = MEMMModel(project).fit(train_set)
        elif method == 'saito':
            model = Saito(project)
        elif method == 'avg':
            model = LTAvg(project)
        else:
            raise Exception('invalid method "%s"' % method)

        # Create a process pool to distribute the prediction.
        process_count = 3
        pool = Pool(processes=process_count)
        step = int(math.ceil(float(len(test_set)) / process_count))
        results = []
        for j in range(0, len(test_set), step):
            meme_ids = test_set[j: j + step]
            res = pool.apply_async(test_meme, (meme_ids, method, model, threshold, trees, all_node_ids, self.verbosity))
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

        # Gather all "meme_id-node_id" pairs as reference set.
        all_nodes = []
        for meme_id in set(test_set).union(set(train_set)):
            all_nodes.extend({'{}-{}'.format(meme_id, node) for node in trees[meme_id].node_ids()})

        # all_res_nodes, all_true_nodes, prp1_list, prp2_list = test_meme(test_set, method, model, threshold, trees)

        # Evaluate total results.
        meas = Validation(all_res_nodes, all_true_nodes, all_nodes)
        prec, rec, f1 = meas.precision(), meas.recall(), meas.f1()
        if self.verbosity > 1:
            logger.info('project %s: %d outputs, %d true, precision = %.3f, recall = %.3f, f1 = %.3f' % (
                project.project_name, len(all_res_nodes), len(all_true_nodes), prec, rec, f1))

        if method in ['saito', 'avg'] and self.verbosity > 1:
            logger.info('prp1 avg = %.3f' % np.mean(np.array(prp1_list)))
            logger.info('prp2 avg = %.3f' % np.mean(np.array(prp2_list)))

        return meas
