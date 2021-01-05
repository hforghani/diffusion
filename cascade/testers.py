import abc
import math
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot

from cascade.asynchroizables import train_memes, test_memes, test_memes_multiproc
from settings import logger, mongodb
from utils.time_utils import time_measure


class ProjectTester(abc.ABC):
    def __init__(self, project, method):
        self.project = project
        self.method = method

        if method in ['aslt', 'avg']:
            # Create dictionary of user id's to their sorted index.
            logger.info('creating dictionary of user ids to their sorted index ...')
            self.user_ids = [u['_id'] for u in mongodb.users.find({}, ['_id']).sort('_id')]
            self.users_map = {self.user_ids[i]: i for i in range(len(self.user_ids))}

    @abc.abstractmethod
    def run(self, thresholds, initial_depth, max_depth):
        """ Run the training, validation, and test stages for the project """

    @time_measure()
    def train(self):
        model = train_memes(self.method, self.project)
        return model

    @time_measure()
    def validate(self, thresholds, initial_depth=0, max_depth=None, multi_processed=False, model=None):
        """
        :param model: trained model, if None the model is trained in test method
        :param thresholds: list of validation thresholds
        :param initial_depth: depth of initial nodes of tree
        :param max_depth: maximum depth of tree to which we want to predict
        :param multi_processed: If True, run multi-processing for each cascade or for each MEMM if MEMM method is used.
        :return:
        """
        _, val_set, _ = self.project.load_sets()
        precs = []
        recs = []
        f1s = []
        fprs = []

        for thr in thresholds:
            logger.info('{0} THRESHOLD = {1:.3f} {0}'.format('=' * 20, thr))

            prec, rec, f1, fpr = self.test(val_set, thr, initial_depth, max_depth, multi_processed, model)

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

    @abc.abstractmethod
    def test(self, test_set, threshold, initial_depth=0, max_depth=None, multi_processed=False, model=None):
        """
        Test by the threshold given.
        :param test_set:
        :param threshold:
        :param initial_depth:
        :param max_depth:
        :param multi_processed:
        :param model:
        :return:
        """

    def __get_mean_results(self, f1s, fprs, precisions, prp1_list, prp2_list, recalls):
        mean_prec = np.array(precisions).mean()
        mean_rec = np.array(recalls).mean()
        mean_fpr = np.array(fprs).mean()
        mean_f1 = np.array(f1s).mean()
        logger.info('{project %s} averages: precision = %.3f, recall = %.3f, f1 = %.3f' % (
            self.project.project_name, mean_prec, mean_rec, mean_f1))
        if self.method in ['aslt', 'avg']:
            logger.info('prp1 avg = %.3f' % np.mean(np.array(prp1_list)))
            logger.info('prp2 avg = %.3f' % np.mean(np.array(prp2_list)))
        # return meas
        return mean_prec, mean_rec, mean_f1, mean_fpr

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


class DefaultTester(ProjectTester):
    def run(self, thresholds, initial_depth, max_depth):
        model = self.train(self.method)
        thr = self.validate(self.method, thresholds, initial_depth, max_depth, model)
        return self.test(self.method, thr, initial_depth, max_depth, model)

    @time_measure()
    def test(self, test_set, threshold, initial_depth=0, max_depth=None, multi_processed=False, model=None):
        # Load training and test sets and cascade trees.
        trees = self.project.load_trees()

        all_node_ids = self.project.get_all_nodes()
        # all_node_ids = self.user_ids

        logger.info('number of memes : %d' % len(test_set))

        precisions, recalls, f1s, fprs, prp1_list, prp2_list = test_memes(test_set, self.method, model, threshold,
                                                                          initial_depth, max_depth, trees,
                                                                          all_node_ids, self.user_ids,
                                                                          self.users_map)

        return self.__get_mean_results(f1s, fprs, precisions, prp1_list, prp2_list, recalls)


class MultiProcTester(ProjectTester):
    def run(self, thresholds, initial_depth, max_depth):
        thr = self.validate(self.method, thresholds, initial_depth, max_depth)
        return self.test(self.method, thr, initial_depth, max_depth)

    @time_measure()
    def test(self, test_set, threshold, initial_depth=0, max_depth=None, multi_processed=False, model=None):
        # Load training and test sets and cascade trees.
        trees = self.project.load_trees()

        all_node_ids = self.project.get_all_nodes()
        # all_node_ids = self.user_ids

        logger.info('number of memes : %d' % len(test_set))

        precisions, recalls, f1s, fprs, prp1_list, prp2_list \
            = self.__test_multi_processed(test_set, threshold, initial_depth, max_depth, trees, all_node_ids)

        return self.__get_mean_results(f1s, fprs, precisions, prp1_list, prp2_list, recalls)

    def __test_multi_processed(self, test_set, threshold, initial_depth, max_depth, trees, all_node_ids):
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
            res = pool.apply_async(test_memes_multiproc,
                                   (meme_ids, self.method, self.project, threshold, initial_depth, max_depth, trees,
                                    all_node_ids,
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
