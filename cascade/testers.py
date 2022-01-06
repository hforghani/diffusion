import abc
import math
import numbers
import os
from multiprocessing import Pool
from typing import Union, Tuple

import numpy as np
from matplotlib import pyplot

import settings
from cascade.asynchroizables import train_cascades, test_cascades, test_cascades_multiproc
from db.managers import DBManager, MEMMManager
from settings import logger
from utils.time_utils import time_measure, Timer


class ProjectTester(abc.ABC):
    def __init__(self, project, method):
        self.project = project
        self.method = method

        if method in ['aslt', 'avg']:
            # Create dictionary of user id's to their sorted index.
            logger.info('creating dictionary of user ids to their sorted index ...')
            db = DBManager(self.project.db).db
            self.user_ids = [u['_id'] for u in db.users.find({}, ['_id']).sort('_id')]
            self.users_map = {self.user_ids[i]: i for i in range(len(self.user_ids))}
        else:
            self.user_ids = None
            self.users_map = None

    @abc.abstractmethod
    def run_validation_test(self, thresholds: Union[list, numbers.Number], initial_depth: int, max_depth: int) \
            -> Tuple[dict, dict, dict, dict]:
        """ Run the training, validation, and test stages for the project """

    @abc.abstractmethod
    def run_test(self, threshold: numbers.Number, initial_depth: int, max_depth: int) \
            -> Tuple[dict, dict, dict, dict]:
        """ Run the training, and test stages for the project """

    @abc.abstractmethod
    def train(self):
        pass

    @time_measure(level='debug')
    def validate(self, val_set, thresholds, initial_depth=0, max_depth=None, model=None):
        """
        :param model: trained model, if None the model is trained in test method
        :param val_set: validation set. list of cascade id's
        :param thresholds: list of validation thresholds
        :param initial_depth: depth of initial nodes of tree
        :param max_depth: maximum depth of tree to which we want to predict
        :return:
        """
        precs, recs, f1s, fprs = self.test(val_set, thresholds, initial_depth, max_depth, model)

        best_thr = max(f1s, key=lambda thr: f1s[thr])
        best_f1 = f1s[best_thr]

        logger.info(f'F1 max = {best_f1} in threshold = {best_thr}')

        self.__save_charts(best_thr, precs, recs, f1s, fprs, thresholds)
        return best_thr

    @abc.abstractmethod
    def test(self, test_set: list, thr: Union[list, numbers.Number], initial_depth: int = 0, max_depth: int = None,
             model=None):
        """
        Test on test set by the threshold(s) given.
        :param test_set: list of cascade ids to test
        :param thr: If it is a number, it is the activation threshold. If it is a list then it is
            considered as list of thresholds. In this case the method returns a list of precisions, recals,
            f1s, FPRs.
        :param initial_depth: the depth to which the nodes considered as initial nodes.
        :param max_depth: the depth to which the prediction will be done.
        :param model: the prediction model instance (an instance of MEMMModel, AsLT, or ...)
        :return: if the threshold is a number, returns tuple of precision, recall, f1, FPR. If a list of thresholds
            is given, it returns tuple of list of precisions, list of recalls, etc. related to the thresholds.
        """

    def _get_mean_results(self, precisions: dict, recalls: dict, f1s: dict, fprs: dict, prp1s: dict,
                          prp2s: dict) -> Tuple[dict, dict, dict, dict]:
        mean_prec = {}
        mean_rec = {}
        mean_fpr = {}
        mean_f1 = {}
        logs = [f'{"threshold":>10}{"precision":>10}{"recall":>10}{"f1":>10}']
        for thr in precisions:
            mean_prec[thr] = np.array(precisions[thr]).mean()
            mean_rec[thr] = np.array(recalls[thr]).mean()
            mean_fpr[thr] = np.array(fprs[thr]).mean()
            mean_f1[thr] = np.array(f1s[thr]).mean()
            logs.append(f'{thr:10.3f}{mean_prec[thr]:10.3f}{mean_rec[thr]:10.3f}{mean_f1[thr]:10.3f}')
            # if self.method in ['aslt', 'avg']:
            #     logger.info('prp1 avg = %.3f', np.mean(np.array(prp1s[thr])))
            #     logger.info('prp2 avg = %.3f', np.mean(np.array(prp2s[thr])))

        logger.info('averages:\n' + '\n'.join(logs))
        return mean_prec, mean_rec, mean_f1, mean_fpr

    def __save_charts(self, best_thr: float, precs: dict, recs: dict, f1s: dict, fprs: dict, thresholds: list):
        precs_list = [precs[thr] for thr in thresholds]
        recs_list = [recs[thr] for thr in thresholds]
        f1s_list = [f1s[thr] for thr in thresholds]
        fprs_list = [fprs[thr] for thr in thresholds]

        pyplot.figure(1)
        pyplot.subplot(221)
        pyplot.plot(thresholds, precs_list)
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.scatter([best_thr], [precs[best_thr]], c='r', marker='o')
        pyplot.title('precision')
        pyplot.subplot(222)
        pyplot.plot(thresholds, recs_list)
        pyplot.scatter([best_thr], [recs[best_thr]], c='r', marker='o')
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.title('recall')
        pyplot.subplot(223)
        pyplot.plot(thresholds, f1s_list)
        pyplot.scatter([best_thr], [f1s[best_thr]], c='r', marker='o')
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.title('F1')
        pyplot.subplot(224)
        pyplot.plot(fprs_list, recs_list)
        pyplot.scatter([fprs[best_thr]], [recs[best_thr]], c='r', marker='o')
        pyplot.title('ROC curve')
        pyplot.axis([0, 1, 0, 1])
        results_path = os.path.join(settings.BASEPATH, 'results')
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        filename = os.path.join(results_path, f'{self.project.project_name}-{self.method}.png')
        pyplot.savefig(filename)
        # pyplot.show()


class DefaultTester(ProjectTester):
    def run_validation_test(self, thresholds, initial_depth, max_depth):
        with Timer('training'):
            model = self.train()

        with Timer('validation & test'):
            _, val_set, test_set = self.project.load_sets()
            logger.info('{0} VALIDATION {0}'.format('=' * 20))
            thr = self.validate(val_set, thresholds, initial_depth, max_depth, model=model)
            logger.info('{0} TEST (threshold = %f) {0}'.format('=' * 20), thr)
            return self.test(test_set, thr, initial_depth, max_depth, model=model)

    def run_test(self, threshold, initial_depth, max_depth):
        with Timer('training'):
            model = self.train()

        with Timer('test'):
            _, val_set, test_set = self.project.load_sets()
            logger.info('{0} TEST (threshold = %f) {0}'.format('=' * 20), threshold)
            return self.test(test_set, threshold, initial_depth, max_depth, model=model)

    @time_measure(level='debug')
    def train(self):
        model = train_cascades(self.method, self.project)
        return model

    @time_measure(level='debug')
    def test(self, test_set: list, thr: Union[list, numbers.Number], initial_depth: int = 0, max_depth: int = None,
             model=None):
        if isinstance(thr, numbers.Number):
            thresholds = [thr]
        else:
            thresholds = thr

        # Load training and test sets and cascade trees.
        trees = self.project.load_trees()
        all_node_ids = self.project.get_all_nodes()

        logger.info('number of cascades : %d' % len(test_set))

        precisions, recalls, f1s, fprs, prp1_list, prp2_list = test_cascades(test_set, self.method, model, thresholds,
                                                                             initial_depth, max_depth, trees,
                                                                             all_node_ids, self.user_ids,
                                                                             self.users_map)
        mean_prec, mean_rec, mean_f1, mean_fpr = self._get_mean_results(precisions, recalls, f1s, fprs, prp1_list,
                                                                        prp2_list)
        if isinstance(thr, numbers.Number):
            return mean_prec[thr], mean_rec[thr], mean_f1[thr], mean_fpr[thr]
        else:
            return mean_prec, mean_rec, mean_f1, mean_fpr


class MultiProcTester(ProjectTester):
    def run_validation_test(self, thresholds, initial_depth, max_depth):
        with Timer('training'):
            # Train the cascades once and save them. Then the trained model is fetched from disk and used at each process.
            # The model is not passed to each process due to pickling size limit.
            if not MEMMManager(self.project, self.method).db_exists():
                self.train()

        with Timer('validation & test'):
            _, val_set, test_set = self.project.load_sets()
            logger.info('{0} VALIDATION {0}'.format('=' * 20))
            thr = self.validate(val_set, thresholds, initial_depth, max_depth)
            logger.info('{0} TEST (threshold = %f) {0}'.format('=' * 20), thr)
            return self.test(test_set, thr, initial_depth, max_depth)

    def run_test(self, threshold, initial_depth, max_depth):
        with Timer('training'):
            # Train the cascades once and save them. Then the trained model is fetched from disk and used at each process.
            # The model is not passed to each process due to pickling size limit.
            if not MEMMManager(self.project, self.method).db_exists():
                self.train()

        with Timer('test'):
            _, val_set, test_set = self.project.load_sets()
            logger.info('{0} TEST (threshold = %f) {0}'.format('=' * 20), threshold)
            return self.test(test_set, threshold, initial_depth, max_depth)

    @time_measure(level='debug')
    def train(self):
        model = train_cascades(self.method, self.project, multi_processed=True)
        return model

    @time_measure(level='debug')
    def test(self, test_set: list, thr: Union[list, numbers.Number], initial_depth: int = 0, max_depth: int = None,
             model=None):
        if isinstance(thr, numbers.Number):
            thresholds = [thr]
        else:
            thresholds = thr

        # Load training and test sets and cascade trees.
        trees = self.project.load_trees()

        all_node_ids = self.project.get_all_nodes()
        # all_node_ids = self.user_ids

        logger.info('number of cascades : %d' % len(test_set))

        precisions, recalls, f1s, fprs, prp1_list, prp2_list \
            = self.__test_multi_processed(test_set, thresholds, initial_depth, max_depth, trees, all_node_ids)

        mean_prec, mean_rec, mean_f1, mean_fpr = self._get_mean_results(precisions, recalls, f1s, fprs, prp1_list,
                                                                        prp2_list)
        if isinstance(thr, numbers.Number):
            return mean_prec[thr], mean_rec[thr], mean_f1[thr], mean_fpr[thr]
        else:
            return mean_prec, mean_rec, mean_f1, mean_fpr

    def __test_multi_processed(self, test_set: list, thresholds: list, initial_depth: int, max_depth: int, trees: dict,
                               all_node_ids: list):
        """
        Create a process pool to distribute the prediction.
        """
        process_count = min(settings.PROCESS_COUNT, len(test_set))
        pool = Pool(processes=process_count)
        step = int(math.ceil(float(len(test_set)) / process_count))
        results = []
        for j in range(0, len(test_set), step):
            cascade_ids = test_set[j: j + step]
            res = pool.apply_async(test_cascades_multiproc,
                                   (cascade_ids, self.method, self.project, thresholds, initial_depth, max_depth,
                                    trees, all_node_ids, self.user_ids, self.users_map))
            results.append(res)

        pool.close()
        pool.join()

        precisions = {thr: [] for thr in thresholds}
        recalls = {thr: [] for thr in thresholds}
        f1s = {thr: [] for thr in thresholds}
        fprs = {thr: [] for thr in thresholds}
        prp1_list = {thr: [] for thr in thresholds}
        prp2_list = {thr: [] for thr in thresholds}

        # Collect results of the processes.
        for res in results:
            prec_subset, rec_subset, f1_subset, fpr_subset, prp1_subset, prp2_subset = res.get()
            for thr in thresholds:
                precisions[thr].extend(prec_subset[thr])
                recalls[thr].extend(rec_subset[thr])
                f1s[thr].extend(f1_subset[thr])
                fprs[thr].extend(fpr_subset[thr])
                prp1_list[thr].extend(prp1_subset[thr])
                prp2_list[thr].extend(prp2_subset[thr])

        return precisions, recalls, f1s, fprs, prp1_list, prp2_list
