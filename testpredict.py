# -*- coding: utf-8 -*-
import argparse
import multiprocessing
from multiprocessing.pool import Pool
import traceback
import time
import math
import numpy as np
from matplotlib import pyplot
# from profilehooks import timecall, profile
from cascade.avg import LTAvg
from cascade.validation import Validation
from cascade.saito import Saito
from cascade.models import Project
from memm.models import MEMMModel
from mln.models import MLN
from mln.file_generators import FileCreator
import settings
from settings import logger, mongodb


def evaluate(initial_tree, res_tree, tree, all_nodes, max_depth=None, verbosity=settings.VERBOSITY):
    # Get predicted and true nodes.
    res_nodes = set(res_tree.node_ids())
    true_nodes = set(tree.node_ids(max_depth=max_depth))
    initial_nodes = set(initial_tree.node_ids())
    res_output = res_nodes - initial_nodes
    true_output = true_nodes - initial_nodes
    if verbosity > 2:
        logger.info('len(all_nodes) = %d', len(all_nodes))

    # Evaluate the result.
    meas = Validation(res_output, true_output, all_nodes)
    return meas, res_output, true_output


def log_trees(tree, res_tree, max_depth=None):
    if max_depth is not None:
        tree = tree.copy(max_depth)
    tree_render = tree.render(digest=True).split('\n')
    res_tree_render = res_tree.render(digest=True).split('\n')
    max_len = max([len(line) for line in tree_render])
    max_lines = max(len(tree_render), len(res_tree_render))
    formatted_line = '{:' + str(max_len + 5) + '}{}'
    logger.info(formatted_line.format('true tree:', 'output tree:'))
    for i in range(max_lines):
        logger.info(formatted_line.format(tree_render[i] if i < len(tree_render) else '',
                                          res_tree_render[i] if i < len(res_tree_render) else ''))


def test_meme(meme_ids, method, model, threshold, initial_depth, max_depth, trees, all_node_ids, user_ids, users_map,
              verbosity=settings.VERBOSITY):
    try:
        prp1_list = []
        prp2_list = []
        precisions = []
        recalls = []
        fprs = []
        f1s = []

        for meme_id in meme_ids:
            tree = trees[meme_id]

            # Copy roots in a new tree.
            initial_tree = tree.copy(initial_depth)

            # Predict remaining nodes.
            logger.info('running prediction with method <%s> on meme <%s>', method, meme_id)
            # TODO: apply max_depth for all methods.
            if method in ['mlnprac', 'mlnalch']:
                res_tree = model.predict(meme_id, initial_tree, threshold=threshold, log=verbosity - 2)
            elif method in ['aslt', 'avg']:
                res_tree = model.predict(initial_tree, threshold=threshold, max_step=max_depth - initial_depth,
                                         user_ids=user_ids, users_map=users_map, log=verbosity - 2)
            else:
                res_tree = model.predict(initial_tree, threshold=threshold, log=verbosity - 2)

            # Evaluate the results.
            meas, res_output, true_output = evaluate(initial_tree, res_tree, tree, all_node_ids, max_depth)

            if method in ['aslt', 'avg']:
                prp = meas.prp(model.probabilities)
                prp1 = prp[0] if prp else 0
                prp2 = prp[1] if len(prp) > 1 else 0
                prp1_list.append(prp1)
                prp2_list.append(prp2)

            prec = meas.precision()
            rec = meas.recall()
            fpr = meas.fpr()
            f1 = meas.f1()
            precisions.append(prec)
            recalls.append(rec)
            fprs.append(fpr)
            f1s.append(f1)

            if verbosity > 1:
                log = 'meme %s: %d outputs, %d true, precision = %.3f, recall = %.3f, , f1 = %.3f' % (
                    meme_id, len(res_output), len(true_output), prec, rec, f1)
                if method in ['aslt', 'avg']:
                    log += ', prp = (%.3f, %.3f, ...)' % (prp1, prp2)
                logger.info(log)
            if verbosity > 2:
                log_trees(tree, res_tree, max_depth)

        return precisions, recalls, f1s, fprs, prp1_list, prp2_list

    except:
        print(traceback.format_exc())
        raise


class Command:
    help = 'Test information diffusion prediction'

    def add_arguments(self, parser):
        parser.add_argument("-p", "--project", type=str, dest="project",
                            help="project name or multiple comma-separated project names")
        parser.add_argument("-m", "--method", type=str, dest="method",
                            choices=['mlnprac', 'mlnalch', 'memm', 'aslt', 'avg'],
                            help="the method by which we want to test")
        parser.add_argument("-t", "--threshold", type=float, dest="threshold",
                            help="the threshold to apply on the method")
        parser.add_argument("-i", "--init-depth", type=int, dest="initial_depth", default=0,
                            help="the maximum depth for the initial nodes")
        parser.add_argument("-d", "--max-depth", type=int, dest="max_depth",
                            help="the maximum depth of cascade prediction")
        parser.add_argument("-a", "--all", action='store_true', dest="all_thresholds", default=False,
                            help="test the method for all threshold values and show the charts")
        parser.add_argument("-u", "--multiprocessed", action='store_true', dest="multi_processed", default=False,
                            help="run tests on multiple processes")
        parser.add_argument("-v", "--verbosity", type=int, dest="verbosity", default=settings.VERBOSITY,
                            help="verbosity level")

    THRESHOLDS_COUNT = 10

    def __init__(self):
        self.verbosity = settings.VERBOSITY
        self.user_ids = None
        self.users_map = None

    def handle(self, args):
        start = time.time()
        project_names = args.project.split(',')
        self.verbosity = args.verbosity
        multi_processed = args.multi_processed

        # Get the method or raise exception.
        method = args.method
        if method is None:
            raise Exception('--method argument required')

        if args.all_thresholds:
            thres_min, thres_max = settings.THRESHOLDS[method]
            step = (thres_max - thres_min) / (self.THRESHOLDS_COUNT - 1)
            thresholds = [step * i + thres_min for i in range(self.THRESHOLDS_COUNT)]
        elif args.threshold is None:
            raise Exception('either --all or --threshold arguments must be given')
        else:
            thresholds = [args.threshold]

        # Log the test configuration.
        logger.info('{0} PROJECT(S) = {1} {0}'.format('=' * 20, project_names))
        logger.info('{0} METHOD = {1} {0}'.format('=' * 20, method))
        logger.info('{0} INITIAL DEPTH = {1} {0}'.format('=' * 20, args.initial_depth))
        logger.info('{0} MAX DEPTH = {1} {0}'.format('=' * 20, args.max_depth))
        logger.info('{0} TESTING ON THRESHOLD(S) {1} {0}'.format('=' * 20, thresholds))

        final_prec = []
        final_recall = []
        final_f1 = []
        final_fpr = []

        if method in ['aslt', 'avg']:
            # Create dictionary of user id's to their sorted index.
            logger.info('creating dictionary of user ids to their sorted index ...')
            self.user_ids = [u['_id'] for u in mongodb.users.find({}, ['_id']).sort('_id')]
            self.users_map = {self.user_ids[i]: i for i in range(len(self.user_ids))}

        models = self.train(method, project_names)

        for thr in thresholds:
            if args.all_thresholds:
                if self.verbosity:
                    logger.info('{0} THRESHOLD = {1:.3f} {0}'.format('=' * 20, thr))
            prec = []
            recall = []
            f1 = []
            fpr = []

            for p_name in project_names:
                model = models[p_name]
                mprec, mrec, mf1, mfpr = self.test(model, method, thr, args.initial_depth, args.max_depth,
                                                   multi_processed)
                prec.append(mprec)
                recall.append(mrec)
                f1.append(mf1)
                fpr.append(mfpr)

            if len(project_names) > 1:
                fprec = np.mean(np.array(prec))
                frecall = np.mean(np.array(recall))
                ff1 = np.mean(np.array(f1))
                ffpr = np.mean(np.array(fpr))
            else:
                fprec = prec[0]
                frecall = recall[0]
                ff1 = f1[0]
                ffpr = fpr[0]

            if self.verbosity:
                logger.info('final precision = %.3f, recall = %.3f, f1 = %.3f, fpr = %.3f', fprec, frecall, ff1, ffpr)

            final_prec.append(fprec)
            final_recall.append(frecall)
            final_f1.append(ff1)
            final_fpr.append(ffpr)

        if self.verbosity:
            logger.info('command done in %.2f min' % ((time.time() - start) / 60))

        if args.all_thresholds:
            # Find the threshold with maximum F1.
            best_ind = int(np.argmax(np.array(final_f1)))
            best_f1, best_thres = final_f1[best_ind], thresholds[best_ind]
            if self.verbosity:
                logger.info('F1 max = %f in threshold = %f' % (best_f1, best_thres))

            self.__display_charts(best_f1, best_ind, best_thres, final_f1, final_fpr, final_prec, final_recall,
                                  thresholds)

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
                model = MEMMModel(project).fit(train_set, log=self.verbosity - 2)
            elif method == 'aslt':
                model = Saito(project)
            elif method == 'avg':
                model = LTAvg(project)
            else:
                raise Exception('invalid method "%s"' % method)
            models[p_name] = model
        return models

    def test(self, model, method, threshold, initial_depth=0, max_depth=None, multi_processed=False):
        # Load training and test sets and cascade trees.
        project = model.project
        train_set, test_set = project.load_train_test()
        trees = project.load_trees(verbosity=self.verbosity - 1)

        all_node_ids = project.get_all_nodes()
        # all_node_ids = self.user_ids

        if self.verbosity > 1:
            logger.info('test set size = %d' % len(test_set))

        if multi_processed:
            precisions, recalls, f1s, fprs, prp1_list, prp2_list = self.__test_multi_processed(test_set, method, model,
                                                                                               threshold, initial_depth,
                                                                                               max_depth, trees,
                                                                                               all_node_ids)
        else:
            precisions, recalls, f1s, fprs, prp1_list, prp2_list = test_meme(test_set, method, model, threshold,
                                                                             initial_depth, max_depth, trees,
                                                                             all_node_ids, self.user_ids,
                                                                             self.users_map,
                                                                             self.verbosity)

        mean_prec = np.array(precisions).mean()
        mean_rec = np.array(recalls).mean()
        mean_fpr = np.array(fprs).mean()
        mean_f1 = np.array(f1s).mean()

        if self.verbosity > 1:
            logger.info('project %s: mean precision = %.3f, mean recall = %.3f, f1 = %.3f' % (
                project.project_name, mean_prec, mean_rec, mean_f1))

        if method in ['aslt', 'avg'] and self.verbosity > 1:
            logger.info('prp1 avg = %.3f' % np.mean(np.array(prp1_list)))
            logger.info('prp2 avg = %.3f' % np.mean(np.array(prp2_list)))

        # return meas
        return mean_prec, mean_rec, mean_f1, mean_fpr

    def __test_multi_processed(self, test_set, method, model, threshold, max_depth, trees, all_node_ids):
        """
        Create a process pool to distribute the prediction.
        """
        # process_count = multiprocessing.cpu_count()
        process_count = 4
        pool = Pool(processes=process_count)
        step = int(math.ceil(float(len(test_set)) / process_count))
        results = []
        for j in range(0, len(test_set), step):
            meme_ids = test_set[j: j + step]
            res = pool.apply_async(test_meme,
                                   (meme_ids, method, model, threshold, initial_depth, max_depth, trees, all_node_ids,
                                    self.user_ids, self.users_map, self.verbosity))
            results.append(res)

        pool.close()
        pool.join()

        prp1_list = []
        prp2_list = []
        precisions = []
        recalls = []
        fprs = []
        f1s = []

        # Collect results of the processes.
        for res in results:
            r1, r2, r3, r4, r5, r6 = res.get()
            precisions.extend(r1)
            recalls.extend(r2)
            fprs.extend(r3)
            recalls.extend(r4)
            prp1_list.extend(r5)
            prp2_list.extend(r6)

        return precisions, recalls, f1s, fprs, prp1_list, prp2_list

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


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
