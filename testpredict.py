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
from utils.time_utils import time_measure, Timer


def evaluate(initial_tree, res_tree, tree, all_nodes, max_depth=None):
    # Get predicted and true nodes.
    res_nodes = set(res_tree.node_ids())
    true_nodes = set(tree.node_ids(max_depth=max_depth))
    initial_nodes = set(initial_tree.node_ids())
    res_output = res_nodes - initial_nodes
    true_output = true_nodes - initial_nodes
    logger.debug('len(all_nodes) = %d', len(all_nodes))

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
    logger.debugv(formatted_line.format('true tree:', 'output tree:'))
    for i in range(max_lines):
        logger.debugv(formatted_line.format(tree_render[i] if i < len(tree_render) else '',
                                            res_tree_render[i] if i < len(res_tree_render) else ''))


def test_meme(meme_ids, method, model, threshold, initial_depth, max_depth, trees, all_node_ids, user_ids, users_map):
    try:
        prp1_list = []
        prp2_list = []
        precisions = []
        recalls = []
        fprs = []
        f1s = []
        max_step = max_depth - initial_depth if max_depth is not None else None

        for meme_id in meme_ids:
            with Timer('getting tree'):
                tree = trees[meme_id]

            # Copy roots in a new tree.
            with Timer('copying tree'):
                initial_tree = tree.copy(initial_depth)

            # Predict remaining nodes.
            with Timer('prediction'):
                logger.info('running prediction with method <%s> on meme <%s>', method, meme_id)
                # TODO: apply max_depth for all methods.
                if method in ['mlnprac', 'mlnalch']:
                    res_tree = model.predict(meme_id, initial_tree, threshold=threshold)
                elif method in ['aslt', 'avg']:
                    res_tree = model.predict(initial_tree, threshold=threshold, max_step=max_step, user_ids=user_ids,
                                             users_map=users_map)
                else:
                    res_tree = model.predict(initial_tree, threshold=threshold, max_step=max_step)

            # Evaluate the results.
            with Timer('evaluating results'):
                meas, res_output, true_output = evaluate(initial_tree, res_tree, tree, all_node_ids, max_depth)

            if method in ['aslt', 'avg']:
                prp = meas.prp(model.probabilities)
                prp1 = prp[0] if prp else 0
                prp2 = prp[1] if len(prp) > 1 else 0
                prp1_list.append(prp1)
                prp2_list.append(prp2)

            with Timer('reporting results'):
                prec = meas.precision()
                rec = meas.recall()
                fpr = meas.fpr()
                f1 = meas.f1()
                precisions.append(prec)
                recalls.append(rec)
                fprs.append(fpr)
                f1s.append(f1)

                log = 'meme %s: %d outputs, %d true, precision = %.3f, recall = %.3f, , f1 = %.3f' % (
                    meme_id, len(res_output), len(true_output), prec, rec, f1)
                if method in ['aslt', 'avg']:
                    log += ', prp = (%.3f, %.3f, ...)' % (prp1, prp2)
                logger.info(log)
                # Notice: This line takes too much execution time:
                # log_trees(tree, res_tree, max_depth)

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
        group = parser.add_mutually_exclusive_group()
        group.add_argument("-t", "--threshold", type=float,
                           help="the threshold to apply on the method")
        group.add_argument("-v", "--validation", action='store_true', default=False,
                           help="learn the best threshold in validation stage")
        parser.add_argument("-c", "--thresh-count", type=int, dest="thresholds_count", default=10,
                            help="in the case the argument --all is given, this argument specifies the number of "
                                 "thresholds to test between min and max thresholds specified in local_settings.py")
        parser.add_argument("-i", "--init-depth", type=int, dest="initial_depth", default=0,
                            help="the maximum depth for the initial nodes")
        parser.add_argument("-d", "--max-depth", type=int, dest="max_depth",
                            help="the maximum depth of cascade prediction")
        parser.add_argument("-u", "--multiprocessed", action='store_true', dest="multi_processed", default=False,
                            help="run tests on multiple processes")

    def __init__(self):
        self.user_ids = None
        self.users_map = None

    def handle(self, args):
        start = time.time()
        project_names = args.project.split(',')
        multi_processed = args.multi_processed

        # Get the method or raise exception.
        method = args.method
        if method is None:
            raise Exception('--method argument required')

        if args.validation:
            thres_min, thres_max = settings.THRESHOLDS[method]
            step = (thres_max - thres_min) / (args.thresholds_count - 1)
            thresholds = [step * i + thres_min for i in range(args.thresholds_count)]
        elif args.threshold is None:
            raise Exception('either --validation or --threshold arguments must be given')
        else:
            thresholds = [args.threshold]

        # Log the test configuration.
        logger.info('{0} DB : {1} {0}'.format('=' * 20, settings.DB_NAME))
        logger.info('{0} PROJECT(S) : {1} {0}'.format('=' * 20, project_names))
        logger.info('{0} METHOD : {1} {0}'.format('=' * 20, method))
        logger.info('{0} INITIAL DEPTH : {1} {0}'.format('=' * 20, args.initial_depth))
        logger.info('{0} MAX DEPTH : {1} {0}'.format('=' * 20, args.max_depth))
        logger.info('{0} TESTING ON THRESHOLD(S) : {1} {0}'.format('=' * 20, thresholds))

        precs = []
        recs = []
        f1s = []
        fprs = []

        if method in ['aslt', 'avg']:
            # Create dictionary of user id's to their sorted index.
            logger.info('creating dictionary of user ids to their sorted index ...')
            self.user_ids = [u['_id'] for u in mongodb.users.find({}, ['_id']).sort('_id')]
            self.users_map = {self.user_ids[i]: i for i in range(len(self.user_ids))}

        for p_name in project_names:
            model = self.train(method, p_name)

            thr = self.validate(model, method, thresholds, args.initial_depth, args.max_depth,
                                multi_processed)

            mprec, mrec, mf1, mfpr = self.test(model, method, thr, args.initial_depth, args.max_depth,
                                               multi_processed)

            logger.info('final precision = %.3f, recall = %.3f, f1 = %.3f, fpr = %.3f', mprec, mrec, mf1, mfpr)

            precs.append(mprec)
            recs.append(mrec)
            f1s.append(mf1)
            fprs.append(mfpr)

        if len(project_names) > 1:
            fprec = np.mean(np.array(precs))
            frec = np.mean(np.array(recs))
            ff1 = np.mean(np.array(f1s))
            ffpr = np.mean(np.array(fprs))
        else:
            fprec = precs[0]
            frec = recs[0]
            ff1 = f1s[0]
            ffpr = fprs[0]

        logger.info('Average of projects: precision = %.3f, recall = %.3f, f1 = %.3f, fpr = %.3f', fprec, frec, ff1,
                    ffpr)

        logger.info('command done in %.2f min' % ((time.time() - start) / 60))

    @time_measure(unit='m')
    def train(self, method, project_name):
        project = Project(project_name)
        # Create and train the model if needed.
        if method == 'mlnprac':
            model = MLN(project, method='edge', format=FileCreator.FORMAT_PRACMLN)
        elif method == 'mlnalch':
            model = MLN(project, method='edge', format=FileCreator.FORMAT_ALCHEMY2)
        elif method == 'memm':
            train_set, _, _ = project.load_sets()
            model = MEMMModel(project).fit(train_set)
        elif method == 'aslt':
            model = Saito(project)
        elif method == 'avg':
            model = LTAvg(project)
        else:
            raise Exception('invalid method "%s"' % method)
        return model

    @time_measure(unit='m')
    def validate(self, model, method, thresholds, initial_depth=0, max_depth=None, multi_processed=False):
        _, val_set, _ = model.project.load_sets()
        precs = []
        recs = []
        f1s = []
        fprs = []

        for thr in thresholds:
            if args.validation:
                logger.info('{0} THRESHOLD = {1:.3f} {0}'.format('=' * 20, thr))

            prec, rec, f1, fpr = self.test(model, method, val_set, thr, initial_depth, max_depth, multi_processed)

            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
            fprs.append(fpr)
            logger.info('precision = %.3f, recall = %.3f, f1 = %.3f, fpr = %.3f', prec, rec, f1, fpr)

        best_ind = int(np.argmax(np.array(f1s)))
        best_f1, best_thr = f1s[best_ind], thresholds[best_ind]

        logger.info(f'F1 max = {best_f1} in threshold = {best_thr}')

        self.__display_charts(best_f1, best_ind, best_thr, f1s, fprs, precs, recs, thresholds)
        return best_thr

    @time_measure(unit='m')
    def test(self, model, method, test_set, threshold, initial_depth=0, max_depth=None, multi_processed=False):
        # Load training and test sets and cascade trees.
        project = model.project
        trees = project.load_trees()

        all_node_ids = project.get_all_nodes()
        # all_node_ids = self.user_ids

        logger.info('number of memes : %d' % len(test_set))

        if multi_processed:
            precisions, recalls, f1s, fprs, prp1_list, prp2_list = self.__test_multi_processed(test_set, method, model,
                                                                                               threshold, initial_depth,
                                                                                               max_depth, trees,
                                                                                               all_node_ids)
        else:
            precisions, recalls, f1s, fprs, prp1_list, prp2_list = test_meme(test_set, method, model, threshold,
                                                                             initial_depth, max_depth, trees,
                                                                             all_node_ids, self.user_ids,
                                                                             self.users_map)

        mean_prec = np.array(precisions).mean()
        mean_rec = np.array(recalls).mean()
        mean_fpr = np.array(fprs).mean()
        mean_f1 = np.array(f1s).mean()

        logger.info('{project %s} averages: precision = %.3f, recall = %.3f, f1 = %.3f' % (
            project.project_name, mean_prec, mean_rec, mean_f1))

        if method in ['aslt', 'avg']:
            logger.info('prp1 avg = %.3f' % np.mean(np.array(prp1_list)))
            logger.info('prp2 avg = %.3f' % np.mean(np.array(prp2_list)))

        # return meas
        return mean_prec, mean_rec, mean_f1, mean_fpr

    def __test_multi_processed(self, test_set, method, model, threshold, initial_depth, max_depth, trees, all_node_ids):
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
                                    self.user_ids, self.users_map))
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

    @staticmethod
    def __display_charts(best_f1, best_ind, best_thres, f1s, fprs, precs, recs, thresholds):
        pyplot.figure(1)
        pyplot.subplot(221)
        pyplot.plot(thresholds, precs)
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.scatter([best_thres], [precs[best_ind]], c='r', marker='o')
        pyplot.title('precision')
        pyplot.subplot(222)
        pyplot.plot(thresholds, recs)
        pyplot.scatter([best_thres], [recs[best_ind]], c='r', marker='o')
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.title('recall')
        pyplot.subplot(223)
        pyplot.plot(thresholds, f1s)
        pyplot.scatter([best_thres], [best_f1], c='r', marker='o')
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.title('F1')
        pyplot.subplot(224)
        pyplot.plot(fprs, recs)
        pyplot.scatter([fprs[best_ind]], [recs[best_ind]], c='r', marker='o')
        pyplot.title('ROC curve')
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.show()


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
