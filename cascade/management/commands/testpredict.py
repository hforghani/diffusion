# -*- coding: utf-8 -*-
import logging
import multiprocessing
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
from profilehooks import timecall, profile

if not apps.ready and not settings.configured:
    django.setup()

from cascade.avg import LTAvg
from cascade.validation import Validation
from cascade.saito import Saito
from cascade.models import Project
from memm.models import MEMMModel
from mln.models import MLN
from mln.file_generators import FileCreator
from crud.models import UserAccount

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


def test_meme(meme_ids, method, model, threshold, trees, all_node_ids, user_ids, users_map,
              verbosity=settings.VERBOSITY):
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
                res_tree = model.predict(meme_id, initial_tree, threshold=threshold, log=verbosity > 2)
            elif method in ['saito', 'avg']:
                res_tree = model.predict(initial_tree, threshold=threshold, user_ids=user_ids, users_map=users_map,
                                         log=verbosity > 2)
            else:
                res_tree = model.predict(initial_tree, threshold=threshold, log=verbosity > 2)

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
        parser.add_argument(
            "-u",
            "--multiprocessed",
            action='store_true',
            dest="multi_processed",
            default=False,
            help="run tests on multiple processes",
        )

    thresholds = {
        'mlnprac': settings.MLNPRAC_THRES,
        'mlnalch': settings.MLNALCH_THRES,
        'memm': settings.MEMM_THRES,
        'saito': settings.ASLT_THRES,
        'avg': settings.LTAVG_THRES
    }

    THRESHOLDS_COUNT = 100

    def __init__(self):
        super(Command, self).__init__()
        self.verbosity = settings.VERBOSITY
        self.user_ids = None
        self.users_map = None

    def handle(self, *args, **options):
        start = time.time()
        project_names = options['project'].split(',')
        self.verbosity = options['verbosity'] if options['verbosity'] is not None else settings.VERBOSITY
        multi_processed = options['multi_processed']

        # Get the method or raise exception.
        method = options['method']
        if method is None:
            raise CommandError('method not specified')

        try:
            settings_thr = self.thresholds[method]
        except KeyError:
            raise CommandError('invalid method "%s"' % method)

        if options['all_thresholds']:
            step = settings_thr / self.THRESHOLDS_COUNT * 2
            thresholds = [step * i for i in range(self.THRESHOLDS_COUNT)]
        else:
            thresholds = [settings_thr]

        final_prec = []
        final_recall = []
        final_f1 = []
        final_fpr = []

        if method in ['saito', 'avg']:
            # Create dictionary of user id's to their sorted index.
            self.user_ids = UserAccount.objects.values_list('id', flat=True).order_by('id')
            self.users_map = {self.user_ids[i]: i for i in range(len(self.user_ids))}

        models = self.train(method, project_names)

        for thr in thresholds:
            if options['all_thresholds']:
                if self.verbosity:
                    logger.info('{0} THRESHOLD = {1} {0}'.format('=' * 20, thr))
            prec = []
            recall = []
            f1 = []
            fpr = []

            for p_name in project_names:
                model = models[p_name]
                measure = self.test(model, method, thr, multi_processed)
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

            if len(project_names) > 1 and self.verbosity:
                logger.info('final precision = %.3f, recall = %.3f, f1 = %.3f, fpr = %.3f', fprec, frecall, ff1, ffpr)

        if options['all_thresholds']:
            # Find the threshold with maximum F1.
            best_ind = int(np.argmax(np.array(final_f1)))
            best_f1, best_thres = final_f1[best_ind], thresholds[best_ind]
            if self.verbosity:
                logger.info('F1 max = %f in threshold = %f' % (best_f1, best_thres))

            self.__display_charts(best_f1, best_ind, best_thres, final_f1, final_fpr, final_prec, final_recall,
                                  thresholds)

        if self.verbosity:
            logger.info('command done in %.2f min' % ((time.time() - start) / 60))

    def train(self, method, project_names):
        models = {}
        for p_name in project_names:
            project = Project(p_name)
            # Create and train the model if needed.
            if method == 'mlnprac':
                model = MLN(project, method='edge', format=FileCreator.FORMAT_PRACMLN)
            elif method == 'mlnalch':
                model = MLN(project, method='edge', format=FileCreator.FORMAT_ALCHEMY2)
            elif method == 'memm':
                train_set, _ = project.load_train_test()
                model = MEMMModel(project).fit(train_set, log=self.verbosity > 2)
            elif method == 'saito':
                model = Saito(project)
            elif method == 'avg':
                model = LTAvg(project)
            else:
                raise Exception('invalid method "%s"' % method)
            models[p_name] = model
        return models

    def test(self, model, method, threshold, multi_processed=False):
        # Load training and test sets and cascade trees.
        project = model.project
        train_set, test_set = project.load_train_test()
        trees = project.load_trees()

        all_node_ids = project.get_all_nodes()
        # all_node_ids = self.user_ids

        if self.verbosity > 1:
            logger.info('test set size = %d' % len(test_set))

        if multi_processed:
            all_res_nodes, all_true_nodes, prp1_list, prp2_list = self.__test_multi_processed(test_set, method, model,
                                                                                              threshold, trees,
                                                                                              all_node_ids)
        else:
            all_res_nodes, all_true_nodes, prp1_list, prp2_list = test_meme(test_set, method, model, threshold, trees,
                                                                            all_node_ids, self.user_ids, self.users_map,
                                                                            self.verbosity)

        # Gather all "meme_id-node_id" pairs as reference set.
        all_nodes = []
        for meme_id in set(test_set).union(set(train_set)):
            all_nodes.extend({'{}-{}'.format(meme_id, node) for node in trees[meme_id].node_ids()})

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

    def __test_multi_processed(self, test_set, method, model, threshold, trees, all_node_ids):
        """
        Create a process pool to distribute the prediction.
        """
        process_count = multiprocessing.cpu_count()
        pool = Pool(processes=process_count)
        step = int(math.ceil(float(len(test_set)) / process_count))
        results = []
        for j in range(0, len(test_set), step):
            meme_ids = test_set[j: j + step]
            res = pool.apply_async(test_meme,
                                   (meme_ids, method, model, threshold, trees, all_node_ids, self.user_ids,
                                    self.users_map, self.verbosity))
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

        return all_res_nodes, all_true_nodes, prp1_list, prp2_list

    def __display_charts(self, best_f1, best_ind, best_thres, final_f1, final_fpr, final_prec, final_recall,
                         thresholds):
        pyplot.figure(1)
        pyplot.subplot(221)
        pyplot.plot(thresholds, final_prec)
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.scatter([best_thres], [final_prec[best_ind]], c='r', marker='o')
        pyplot.title('precision')
        pyplot.subplot(222)
        pyplot.plot(thresholds, final_recall)
        pyplot.scatter([best_thres], [final_recall[best_ind]], c='r', marker='o')
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.title('recall')
        pyplot.subplot(223)
        pyplot.plot(thresholds, final_f1)
        pyplot.scatter([best_thres], [best_f1], c='r', marker='o')
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.title('F1')
        pyplot.subplot(224)
        pyplot.plot(final_fpr, final_recall)
        pyplot.scatter([final_fpr[best_ind]], [final_recall[best_ind]], c='r', marker='o')
        pyplot.title('ROC curve')
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.show()
