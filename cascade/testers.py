import math
import os
import typing
from concurrent.futures import ProcessPoolExecutor

import scipy
from bson import ObjectId
from matplotlib import pyplot
from networkx import DiGraph
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import settings
from cascade.asynchroizables import test_cascades, evaluate_nodes, evaluate_edges
from cascade.metric import Metric
from cascade.models import Project
from diffusion.aslt import AsLT
from diffusion.avg import LTAvg
from diffusion.ctic import CTIC
from diffusion.enum import Criterion
from diffusion.ic_models import DAIC, EMIC
from diffusion.models import DiffusionModel
from seq_labeling.crf_models import *
from seq_labeling.memm_models import *
from mln.models import TuffyICMLNModel
from seq_labeling.unified_models import UnifiedMRFModel
from settings import logger
from utils.time_utils import time_measure, Timer, TimeUnit

METHOD_MODEL_MAP = {
    Method.ASLT: AsLT,
    Method.CTIC: CTIC,
    Method.AVG: LTAvg,
    Method.EMIC: EMIC,
    Method.DAIC: DAIC,

    Method.LONG_MEMM: LongMEMMModel,
    Method.BIN_MEMM: BinMEMMModel,
    Method.TD_MEMM: TDMEMMModel,
    Method.MULTI_STATE_LONG_MEMM: MultiStateLongMEMMModel,
    Method.MULTI_STATE_BIN_MEMM: MultiStateBinMEMMModel,
    Method.MULTI_STATE_TD_MEMM: MultiStateTDMEMMModel,
    Method.PARENT_SENS_TD_MEMM: ParentSensTDMEMMModel,
    Method.LONG_PARENT_SENS_TD_MEMM: LongParentSensTDMEMMModel,

    Method.LONG_CRF: CRFModel,
    Method.BIN_CRF: BinCRFModel,
    Method.TD_CRF: TDCRFModel,
    Method.MULTI_STATE_LONG_CRF: MultiStateCRFModel,
    Method.MULTI_STATE_BIN_CRF: MultiStateBinCRFModel,
    Method.MULTI_STATE_TD_CRF: MultiStateTDCRFModel,

    Method.MLN_TUFFY: TuffyICMLNModel,

    Method.UNI_MRF: UnifiedMRFModel
}


def trees_f1_scorer(true_trees, pred_trees, initial_depth, max_depth, graph, criterion=Criterion.NODES):
    f1_values = []

    for i in range(len(true_trees)):
        true_tree = true_trees[i]
        pred_tree = pred_trees[i]
        initial_tree = true_tree.copy(initial_depth)

        if criterion == Criterion.NODES:
            all_node_ids = list(graph.nodes())
            meas, _, _ = evaluate_nodes(initial_tree, pred_tree, true_tree, all_node_ids, max_depth)
        else:
            all_edges = set(graph.edges())
            meas, _, _ = evaluate_edges(initial_tree, pred_tree, true_tree, all_edges, max_depth)
        f1_values.append(meas.f1)

    f1 = np.mean(np.array(f1_values))
    return f1


class ProjectTester(abc.ABC):
    multi_processed = None  # Must be overriden in subclasses.

    def __init__(self, project: Project, method: Method, criterion: Criterion = Criterion.NODES, eco: bool = False):
        self.project = project
        self.method = method
        self.criterion = criterion
        self.eco = eco

    def run(self, initial_depth: int, max_depth: int, n_iter: int = None, **kwargs) -> tuple:
        """ Run cross-validation for the project """
        tunables = {key: kwargs[key] for key in kwargs if isinstance(kwargs[key], tuple)}
        nontunables = {key: kwargs[key] for key in kwargs if key not in tunables}
        logger.debug('tunables = %s', tunables)
        logger.debug('nontunables = %s', nontunables)
        train_set, test_set = self.project.load_sets()

        if tunables:
            if list(tunables.keys()) == ['threshold']:  # There is just one hyperparameter "threshold" to tune.
                logger.info('{0} VALIDATION {0}'.format('=' * 20))
                best_params = self._tune_threshold(initial_depth, max_depth, n_iter, **kwargs)
            else:
                best_params = self._tune_params(initial_depth, max_depth, tunables, nontunables, n_iter)
            logger.info('best_params = %s', best_params)
        else:
            best_params = kwargs
            logger.info('params = %s', kwargs)

        logger.info('{0} TRAINING {0}'.format('=' * 20))
        model = self.train(train_set, eco=self.eco, **best_params)
        logger.info('{0} TEST {0}'.format('=' * 20))
        graph = self.project.load_or_extract_graph(train_set)
        mean_res, res_eval, res_trees = self.test(test_set, model, graph, initial_depth, max_depth, **best_params)
        model.clean_temp_files()
        logger.debug('type(mean_res) = %s', type(mean_res))
        return mean_res, res_eval, res_trees

    def _get_model(self, initial_depth=0, max_depth=None, **params):
        max_step = max_depth - initial_depth if max_depth else None
        if self.method in METHOD_MODEL_MAP:
            model_clazz = METHOD_MODEL_MAP[self.method]
            model = model_clazz(initial_depth=initial_depth, max_step=max_step, **params)
        else:
            raise Exception('invalid method "%s"' % self.method.value)
        return model

    def train(self, train_set, eco, **kwargs):
        model = self._get_model(**kwargs)
        logger.info('loading trees ...')
        trees = self.project.load_trees()
        # TODO: Is is necessary to apply initial_depth and max_depth to trees?
        train_trees = [trees[cid] for cid in train_set]
        with Timer(f'training of method [{self.method.value}] on project [{self.project.name}]', unit=TimeUnit.SECONDS):
            model.fit(train_set, train_trees, self.project, multi_processed=self.multi_processed, eco=eco, **kwargs)
        return model

    @time_measure('debug')
    def test(self, test_set: list, model: DiffusionModel, graph: DiGraph, initial_depth: int = 0, max_depth: int = None,
             **params) -> tuple:
        """
        Test on test set by the threshold(s) given.
        :param test_set: list of cascade ids to test
        :param initial_depth: the depth to which the nodes considered as initial nodes.
        :param max_depth: the depth to which the prediction will be done.
        :param model: the prediction model instance (an instance of MEMMModel, AsLT, or ...)
        :return: if the threshold is a number, returns Metric instance. If a list of thresholds
            is given, it returns a dict of thresholds to list of Metric instances related to the thresholds.
        """
        # Load cascade trees and graph.
        trees = self.project.load_trees()

        logger.info('number of cascades : %d' % len(test_set))

        if all([initial_depth >= trees[cid].depth for cid in test_set]):
            logger.warning(
                'no average results since the initial depth is more than or equal to the depths of all trees')
            return None

        res_eval, res_trees = self._do_test(test_set, model, graph, trees, initial_depth, max_depth, **params)

        if isinstance(res_eval, list):  # It is on test stage
            self._log_cascades_results(test_set, res_eval)

        logger.debug('type(res_eval) = %s', type(res_eval))
        if isinstance(res_eval, list):  # It is on test stage
            mean_res = self._get_mean_results(res_eval)
        else:  # It is on validation stage and results is a dict.
            mean_res = self._get_mean_results_dict(res_eval)

        return mean_res, res_eval, res_trees

    @time_measure(level='info')
    def _tune_threshold(self, initial_depth, max_depth, n_iter, threshold, **params):
        """
        The tunable parameters are related to the test step (indeed they are hyperparameters and are not used
        in the training step).
        :param initial_depth: depth of initial nodes of tree
        :param max_depth: maximum depth of tree to which we want to predict
        :return:
        """
        folds_num = 3
        folds = self.get_cross_val_folds(folds_num)
        if isinstance(threshold, tuple):
            start, end = threshold
            threshold = [round(val, 5) for val in np.arange(start, end, (end - start) / (n_iter - 1))]
        results = {}

        # for each fold, train on the others and test on that fold.
        for i in range(folds_num):
            val_set = folds[i]
            train_set = reduce(lambda x, y: x + y, folds[:i] + folds[i + 1:], [])
            logger.info('{0} TRAINING with fold {1} left {0}'.format('=' * 20, i + 1))
            model = self.train(train_set, eco=False, threshold=threshold, **params)
            logger.info('{0} VALIDATION {0}'.format('=' * 20))
            graph = self.project.load_or_extract_graph(train_set)
            fold_res, _, _ = self.test(val_set, model, graph, initial_depth, max_depth, threshold=threshold, **params)
            model.clean_temp_files()
            del model
            for thr in threshold:
                results.setdefault(thr, [])
                results[thr].append(fold_res[thr].f1)

        # logger.debug('results = %s', results)
        mean_res = {thr: np.mean(np.array(results[thr])) for thr in results}
        logger.debug('mean_res = %s', pprint.pformat(mean_res))
        best_thr = max(mean_res, key=lambda thr: mean_res[thr])
        best_params = params.copy()
        best_params['threshold'] = best_thr
        return best_params

    def _tune_params(self, initial_depth, max_depth, tunables, nontunables, n_iter):
        graph = self.project.load_or_extract_graph()
        f1_scorer = make_scorer(trees_f1_scorer,            # TODO: Pass only nodes or edges instead of graph.
                                initial_depth=initial_depth,
                                max_depth=max_depth,
                                graph=graph,
                                criterion=self.criterion)

        model = self._get_model(initial_depth, max_depth, **nontunables)
        if isinstance(model, CRFModel):
            model.keep_temp_files = False

        # search
        folds_num = 3
        n_jobs = settings.RSCV_WORKERS if self.multi_processed else 1
        # scv = GridSearchCV(model, tunables, cv=folds_num, verbose=2, n_jobs=n_jobs, scoring=f1_scorer, refit=False)
        distributions = {param: scipy.stats.uniform(loc=values[0], scale=values[1] - values[0]) for param, values in
                         tunables.items()}
        logger.info('starting randomized search cross-validation ...')
        scv = RandomizedSearchCV(model, distributions, cv=folds_num, n_iter=n_iter, verbose=3, n_jobs=n_jobs,
                                 refit=False, scoring=f1_scorer)

        train_set, test_set = self.project.load_sets()
        trees = self.project.load_trees()
        # TODO: Is it necessary to apply initial_depth and max_depth to trees?
        train_trees = [trees[cid] for cid in train_set]
        scv.fit(train_set, train_trees, project=self.project)
        best_params = scv.best_params_.copy()
        best_params.update(nontunables)

        return best_params

    def get_cross_val_folds(self, folds_num):
        train_set, test_set = self.project.load_sets()
        fold_size = math.ceil(len(train_set) / folds_num)
        folds = [train_set[i: i + fold_size] for i in range(0, len(train_set), fold_size)]
        return folds

    @abc.abstractmethod
    def _do_test(self, test_set: list, model, graph: DiGraph, trees: dict, initial_depth: int = 0,
                 max_depth: int = None, **params) -> typing.Union[dict, list]:
        pass

    def _get_mean_results_dict(self, results: dict) -> dict:
        """
        :param results: dictionary of thresholds to list of Metric instances
        :return: dictionary of thresholds to Metric instances containing average of precision, recall, f1, and fpr
        """
        mean_res = {}
        logs = [
            'averages:',
            f'{"threshold":<10}{"precision":<10}{"recall":<10}{"f1":<10}'
        ]
        for thr in results:
            mean_metric = Metric([], [])
            prec = np.array([m.precision for m in results[thr] if m is not None]).mean()
            rec = np.array([m.recall for m in results[thr] if m is not None]).mean()
            fpr = np.array([m.fpr for m in results[thr] if m is not None]).mean()
            f1 = np.array([m.f1 for m in results[thr] if m is not None]).mean()
            mean_metric.set(prec, rec, f1, fpr)
            mean_res[thr] = mean_metric
            logs.append(f'{thr:<10.3f}{prec:<10.3f}{rec:<10.3f}{f1:<10.3f}')

        logger.debug('\n'.join(logs))
        return mean_res

    def _get_mean_results(self, results: list) -> Metric:
        """
        :param results: list of Metric instances
        :return: Metric instance containing average of precision, recall, f1, and fpr
        """
        logs = [
            'averages:',
            f'{"precision":<10}{"recall":<10}{"f1":<10}'
        ]
        mean_metric = Metric([], [])
        prec = np.array([m.precision for m in results if m is not None]).mean()
        rec = np.array([m.recall for m in results if m is not None]).mean()
        fpr = np.array([m.fpr for m in results if m is not None]).mean()
        f1 = np.array([m.f1 for m in results if m is not None]).mean()
        mean_metric.set(prec, rec, f1, fpr)
        logs.append(f'{prec:<10.3f}{rec:<10.3f}{f1:<10.3f}')

        logger.info('\n'.join(logs))
        return mean_metric

    def _log_cascades_results(self, test_set: list, results: typing.List[Metric]):
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
            if results[i]:
                cells = (test_set[i], results[i].precision, results[i].recall, results[i].f1)
            else:
                cells = (test_set[i], None, None, None)
            row = ''.join(format_cell(cell) for cell in cells)
            logs.append(row)

        logger.info('\n'.join(logs))

    def __save_charts(self, best_thr: float, results: dict, thresholds: list, initial_depth: int, max_depth: int):
        precs_list = [results[thr].precision for thr in thresholds]
        recs_list = [results[thr].recall for thr in thresholds]
        f1s_list = [results[thr].f1 for thr in thresholds]
        fprs_list = [results[thr].fpr for thr in thresholds]
        best_res = results[best_thr]

        pyplot.figure()
        pyplot.subplot(221)
        pyplot.plot(thresholds, precs_list)
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.scatter([best_thr], [best_res.precision], c='r', marker='o')
        pyplot.title('precision')
        pyplot.subplot(222)
        pyplot.plot(thresholds, recs_list)
        pyplot.scatter([best_thr], [best_res.recall], c='r', marker='o')
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.title('recall')
        pyplot.subplot(223)
        pyplot.plot(thresholds, f1s_list)
        pyplot.scatter([best_thr], [best_res.f1], c='r', marker='o')
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.title('F1')
        pyplot.subplot(224)
        pyplot.plot(fprs_list, recs_list)
        pyplot.scatter([best_res.fpr], [best_res.recall], c='r', marker='o')
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
    multi_processed = False

    def _do_test(self, test_set, model, graph, trees, initial_depth=0, max_depth=None, **params):
        return test_cascades(test_set, self.method, model, initial_depth, max_depth, self.criterion, trees,
                             graph, **params)


class MultiProcTester(ProjectTester):
    multi_processed = True

    def _do_test(self, test_set, model, graph, trees, initial_depth=0, max_depth=None, **params):
        """
        Create a process pool to distribute the prediction.
        Side effect: Will clear the dictionary "tree" to free RAM.
        """
        max_workers = settings.TEST_WORKERS
        step = max(1, min(100, len(test_set) // max_workers))
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, len(test_set), step):
                cascades_i = test_set[i:i + step]
                trees_i = {cid: trees[cid] for cid in cascades_i}
                f = executor.submit(test_cascades, cascades_i, self.method, model, initial_depth, max_depth,
                                    self.criterion, trees_i, graph, **params)
                futures.append(f)
        got_results = [f.result() for f in futures]

        result_eval = [res[0] for res in got_results]
        result_trees = [res[1] for res in got_results]
        del got_results
        if isinstance(result_eval[0], list):  # It is on test stage.
            merged_res_eval = reduce(lambda x, y: x + y, result_eval, [])
            merged_res_trees = reduce(lambda x, y: x + y, result_trees, [])
        else:  # It is on validation stage.
            merged_res_eval = reduce(lambda x, y: {key: x.get(key, []) + y[key] for key in y}, result_eval, {})
            merged_res_trees = None  # Result trees are not returned on validation stage.
        return merged_res_eval, merged_res_trees
