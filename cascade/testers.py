import numbers
import os
from multiprocessing import Pool
from typing import Union, Tuple

from matplotlib import pyplot

from cascade.asynchroizables import test_cascades
from cascade.models import Project
from diffusion.aslt import AsLT
from diffusion.avg import LTAvg
from diffusion.ctic import CTIC
from diffusion.enum import Criterion
from diffusion.ic_models import DAIC, EMIC
from memm.models import *
from mln.file_generators import FileCreator
from mln.models import MLN
from settings import logger
from utils.time_utils import time_measure, Timer, TimeUnit


class ProjectTester(abc.ABC):
    def __init__(self, project: Project, method: Method, criterion: Criterion = Criterion.NODES, eco: bool = False):
        self.project = project
        self.method = method
        self.model = None
        self.criterion = criterion
        self.eco = eco

    def run_validation_test(self, thresholds: Union[list, numbers.Number], initial_depth: int, max_depth: int, **kwargs) \
            -> Tuple[float, float, float, float, float]:
        """ Run the training, validation, and test stages for the project """
        if self.model is None:
            self.model = self.train(**kwargs)

        with Timer('validation & test'):
            _, val_set, test_set = self.project.load_sets()
            logger.info('{0} VALIDATION {0}'.format('=' * 20))
            thr = self.validate(val_set, thresholds, initial_depth, max_depth, model=self.model)
            logger.info('{0} TEST (threshold = %f) {0}'.format('=' * 20), thr)
            precision, recall, f1, fpr = self.test(test_set, thr, initial_depth, max_depth, model=self.model)
            return precision, recall, f1, fpr, thr

    def run_test(self, threshold: numbers.Number, initial_depth: int, max_depth: int, **kwargs) \
            -> Tuple[float, float, float, float]:
        """ Run the training, and test stages for the project """
        if self.model is None:
            self.model = self.train(**kwargs)

        with Timer('test'):
            _, val_set, test_set = self.project.load_sets()
            logger.info('{0} TEST (threshold = %f) {0}'.format('=' * 20), threshold)
            return self.test(test_set, threshold, initial_depth, max_depth, model=self.model)

    def _do_train(self, multi_processed, **kwargs):
        with Timer(f'training of method [{self.method.value}] on project [{self.project.name}]', unit=TimeUnit.SECONDS):
            model_classes = {
                Method.ASLT: AsLT,
                Method.CTIC: CTIC,
                Method.AVG: LTAvg,
                Method.EMIC: EMIC,
                Method.DAIC: DAIC,
                Method.BIN_MEMM: BinMEMMModel,
                Method.TD_MEMM: TDMEMMModel,
                Method.REDUCED_TD_MEMM: ReducedTDMEMMModel,
                Method.REDUCED_BIN_MEMM: ReducedBinMEMMModel,
                Method.PARENT_SENS_TD_MEMM: ParentSensTDMEMMModel,
                Method.LONG_PARENT_SENS_TD_MEMM: LongParentSensTDMEMMModel,
                Method.REDUCED_FULL_TD_MEMM: ReducedFullTDMEMMModel,
                Method.TD_EDGE_MEMM: TDEdgeMEMMModel,
            }
            # Create and train the model if needed.
            if self.method == Method.MLN_PRAC:
                model = MLN(self.project, method='edge', format=FileCreator.FORMAT_PRACMLN)
            elif self.method == Method.MLN_ALCH:
                model = MLN(self.project, method='edge', format=FileCreator.FORMAT_ALCHEMY2)
            elif self.method in model_classes:
                model_clazz = model_classes[self.method]
                model = model_clazz(self.project).fit(multi_processed=multi_processed, eco=self.eco, **kwargs)
            else:
                raise Exception('invalid method "%s"' % self.method.value)
            return model

    @abc.abstractmethod
    def train(self, **kwargs):
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

        self.__save_charts(best_thr, precs, recs, f1s, fprs, thresholds, initial_depth, max_depth)
        return best_thr

    def test(self, test_set: list, thr: Union[list, numbers.Number], initial_depth: int = 0, max_depth: int = None,
             model=None) -> tuple:
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
        if isinstance(thr, numbers.Number):
            thresholds = [thr]
        else:
            thresholds = thr

        # Load cascade trees and graph.
        trees = self.project.load_trees()
        graph = self.project.load_or_extract_graph()

        logger.info('number of cascades : %d' % len(test_set))

        precisions, recalls, f1s, fprs, prp1_list, prp2_list = self._do_test(test_set, thresholds, graph, trees,
                                                                             initial_depth, max_depth, model)

        if isinstance(thr, numbers.Number):  # It is in test stage
            self._log_cascades_results(test_set, precisions[thr], recalls[thr], f1s[thr])

        mean_prec, mean_rec, mean_f1, mean_fpr = self._get_mean_results(precisions, recalls, f1s, fprs, prp1_list,
                                                                        prp2_list)
        if isinstance(thr, numbers.Number):  # It is in test stage
            return mean_prec[thr], mean_rec[thr], mean_f1[thr], mean_fpr[thr]
        else:  # It is in validation stage
            return mean_prec, mean_rec, mean_f1, mean_fpr

    @abc.abstractmethod
    def _do_test(self, test_set: list, thresholds: list, graph: DiGraph, trees: dict, initial_depth: int = 0,
                 max_depth: int = None, model=None, ) \
            -> Tuple[dict, dict, dict, dict, dict, dict]:
        pass

    def _get_mean_results(self, precisions: dict, recalls: dict, f1s: dict, fprs: dict, prp1s: dict,
                          prp2s: dict) -> Tuple[dict, dict, dict, dict]:
        if all([val is None for val in next(iter(precisions.values()))]):
            logger.info('no average results since the initial depth is more than or equal to the depths of all trees')
            dic = {thr: None for thr in precisions}
            return (dic,) * 4

        mean_prec = {}
        mean_rec = {}
        mean_fpr = {}
        mean_f1 = {}
        logs = [
            'averages:',
            f'{"threshold":<10}{"precision":<10}{"recall":<10}{"f1":<10}'
        ]
        for thr in precisions:
            mean_prec[thr] = np.array([val for val in precisions[thr] if val is not None]).mean()
            mean_rec[thr] = np.array([val for val in recalls[thr] if val is not None]).mean()
            mean_fpr[thr] = np.array([val for val in fprs[thr] if val is not None]).mean()
            mean_f1[thr] = np.array([val for val in f1s[thr] if val is not None]).mean()
            logs.append(f'{thr:<10.3f}{mean_prec[thr]:<10.3f}{mean_rec[thr]:<10.3f}{mean_f1[thr]:<10.3f}')
            # if self.method in [Method.MLN_PRAC, Method.MLN_ALCH]:
            #     mean_prp1[thr] = np.array([val for val in prp1s[thr] if val is not None]).mean()
            #     mean_prp2[thr] = np.array([val for val in prp2s[thr] if val is not None]).mean()

        if len(precisions) == 1:
            logger.info('\n'.join(logs))
        else:
            logger.debug('\n'.join(logs))
        return mean_prec, mean_rec, mean_f1, mean_fpr

    def _log_cascades_results(self, test_set: list, precisions: list, recalls: list, f1s: list):
        def format_cell(value):
            if value is None:
                return f'{"-":<10}'
            elif isinstance(value, ObjectId):
                return f'{str(value):<30}'
            else:
                return f'{value:<10.3f}'

        logs = ['results on test set:',
                f'{"cascade id":<30}{"precision":<10}{"recall":<10}{"f1":<10}'
                ]
        for i in range(len(test_set)):
            logs.append(
                format_cell(test_set[i]) + format_cell(precisions[i]) + format_cell(recalls[i]) + format_cell(f1s[i]))

        logger.info('\n'.join(logs))

    def __save_charts(self, best_thr: float, precs: dict, recs: dict, f1s: dict, fprs: dict, thresholds: list,
                      initial_depth: int, max_depth: int):
        precs_list = [precs[thr] for thr in thresholds]
        recs_list = [recs[thr] for thr in thresholds]
        f1s_list = [f1s[thr] for thr in thresholds]
        fprs_list = [fprs[thr] for thr in thresholds]

        pyplot.figure()
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
        results_path = os.path.join(settings.BASE_PATH, 'results')
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        filename = os.path.join(results_path,
                                f'{self.project.name}-{self.method}-{initial_depth}-{max_depth}.png')
        pyplot.savefig(filename)
        # pyplot.show()


class DefaultTester(ProjectTester):
    def train(self, **kwargs):
        return self._do_train(multi_processed=False)

    def _do_test(self, test_set, thresholds, graph, trees, initial_depth=0, max_depth=None, model=None):
        return test_cascades(test_set, self.method, model, thresholds, initial_depth, max_depth, self.criterion, trees,
                             graph)


class MultiProcTester(ProjectTester):
    def train(self, **kwargs):
        return self._do_train(multi_processed=True)

    def _do_test(self, test_set, thresholds, graph, trees, initial_depth=0, max_depth=None, model=None):
        """
        Create a process pool to distribute the prediction.
        """
        process_count = min(settings.PROCESS_COUNT, len(test_set))
        pool = Pool(processes=process_count)
        step = int(math.ceil(float(len(test_set)) / process_count))
        results = []
        for j in range(0, len(test_set), step):
            cascade_ids = test_set[j: j + step]
            res = pool.apply_async(test_cascades,
                                   (cascade_ids, self.method, self.model, thresholds, initial_depth, max_depth,
                                    self.criterion, trees, graph))
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
