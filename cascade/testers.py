import numbers
import os
from multiprocessing import Pool
from typing import Union, Any

from matplotlib import pyplot
from sklearn.metrics import f1_score

from cascade.asynchroizables import test_cascades
from cascade.metric import Metric
from cascade.models import Project, ParamTypes
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
            -> Metric:
        """ Run the training, validation, and test stages for the project """
        if self.model is None:
            self.model = self.train(**kwargs)

        with Timer('validation & test'):
            _, val_set, test_set = self.project.load_sets()
            logger.info('{0} VALIDATION {0}'.format('=' * 20))
            thr = self.validate(val_set, thresholds, initial_depth, max_depth, model=self.model)
            if isinstance(thr, numbers.Number):
                logger.info('{0} TEST (threshold = %f) {0}'.format('=' * 20), thr)
            else:
                logger.info('{0} TEST {0}'.format('=' * 20))
            return self.test(test_set, thr, initial_depth, max_depth, model=self.model)

    def run_test(self, threshold: numbers.Number, initial_depth: int, max_depth: int, **kwargs) -> Metric:
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
                Method.THR_REDUCED_TD_MEMM: ThrTDMEMMModel,
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
    def __default_validate(self, val_set, thresholds, initial_depth=0, max_depth=None, model=None):
        """
        :param model: trained model, if None the model is trained in test method
        :param val_set: validation set. list of cascade id's
        :param thresholds: list of validation thresholds
        :param initial_depth: depth of initial nodes of tree
        :param max_depth: maximum depth of tree to which we want to predict
        :return:
        """
        results = self.test(val_set, thresholds, initial_depth, max_depth, model)

        if results is None:
            return None

        best_thr = max(results, key=lambda thr: results[thr].f1())
        best_f1 = results[best_thr].f1()

        logger.info(f'F1 max = {best_f1} in threshold = {best_thr}')

        self.__save_charts(best_thr, results, thresholds, initial_depth, max_depth)
        return best_thr

    @time_measure(level='debug')
    def validate(self, val_set, thresholds, initial_depth=0, max_depth=None, model=None):
        if self.method == Method.THR_REDUCED_TD_MEMM:
            return self.__validate_multi_thr(val_set, thresholds, model)
        else:
            return self.__default_validate(val_set, thresholds, initial_depth, max_depth, model)

    def __validate_multi_thr(self, val_set, thresholds, model):
        try:
            best_thresholds = self.project.load_param('thresholds', ParamTypes.JSON)
            best_thresholds = {ObjectId(node_id): thr for node_id, thr in best_thresholds.items()}
            logger.info('thresholds loaded')
        except FileNotFoundError:
            evidences = ReducedTDMEMMModel(self.project).prepare_evidences(val_set,
                                                                           multi_processed=self._evid_multiprocessed(),
                                                                           eco=False)
            graph = self.project.load_or_extract_graph()
            best_thresholds = {}
            i = 0
            logger.info('selecting thresholds for %d nodes ...', graph.number_of_nodes())

            for node_id in graph.nodes():
                if node_id in evidences:
                    sequences = evidences[node_id]['sequences']
                    memm = model.get_memm(node_id)
                    if memm:
                        obs, states = self.__sequences_to_obs_states(sequences, memm)
                        n_samples = obs.shape[0]
                        if states.any():
                            f1s = {}
                            probs = memm.multi_prob(obs)
                            # logger.debugv('obs = \n%s', obs)
                            # logger.debugv('states = \n%s', np.squeeze(states))
                            # logger.debugv('probs = \n%s', probs)
                            for thr in thresholds:
                                pred_states = probs >= thr
                                f1 = f1_score(states, pred_states)
                                f1s[thr] = f1
                                # logger.debugv('thr = %f, f1 = %f, pred_states = %s', thr, f1, np.squeeze(pred_states))
                            best_thr = max(f1s, key=lambda thr: f1s[thr])
                            best_thresholds[node_id] = best_thr
                            logger.debugv('best threshold for %s with %d obs : %f', node_id, n_samples, best_thr)
                i += 1
                if i % 100 == 0:
                    logger.debug('threshold selected for %d nodes', i)

            # Set the threshold of any node absent in the validation set equal to the mean thrshold.
            mean_thr = float(np.mean(np.array(list(best_thresholds.values()))))
            # logger.debugv('mean_thr = %s', mean_thr)
            for node_id in graph.nodes():
                best_thresholds.setdefault(node_id, mean_thr)

            self.project.save_param({str(node_id): thr for node_id, thr in best_thresholds.items()}, 'thresholds',
                                    ParamTypes.JSON)
        return best_thresholds

    def test(self, test_set: list, thr: Any, initial_depth: int = 0, max_depth: int = None,
             model=None) -> typing.Union[Metric, dict, None]:
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
        # Load cascade trees and graph.
        trees = self.project.load_trees()
        graph = self.project.load_or_extract_graph()

        logger.info('number of cascades : %d' % len(test_set))

        if all([initial_depth >= trees[cid].depth for cid in test_set]):
            logger.warning(
                'no average results since the initial depth is more than or equal to the depths of all trees')
            return None

        if isinstance(thr, numbers.Number):  # It is on test stage.
            thr = [thr]

        results = self._do_test(test_set, thr, graph, trees, initial_depth, max_depth, model)

        if isinstance(results, list):  # It is in test stage
            self._log_cascades_results(test_set, results)

        if isinstance(results, list):  # It is in test stage
            mean_res = self._get_mean_results(results)
        else:
            mean_res = self._get_mean_results_dict(results)
        return mean_res

    @abc.abstractmethod
    def _do_test(self, test_set: list, thresholds: Any, graph: DiGraph, trees: dict, initial_depth: int = 0,
                 max_depth: int = None, model=None, ) -> typing.Union[dict, list]:
        pass

    def _get_mean_results_dict(self, results: dict) -> dict:
        mean_res = {}
        logs = [
            'averages:',
            f'{"threshold":<10}{"precision":<10}{"recall":<10}{"f1":<10}'
        ]
        for thr in results:
            mean_metric = Metric([], [])
            prec = np.array([m.precision() for m in results[thr] if m is not None]).mean()
            rec = np.array([m.recall() for m in results[thr] if m is not None]).mean()
            fpr = np.array([m.fpr() for m in results[thr] if m is not None]).mean()
            f1 = np.array([m.f1() for m in results[thr] if m is not None]).mean()
            mean_metric.set(prec, rec, f1, fpr)
            mean_res[thr] = mean_metric
            logs.append(f'{thr:<10.3f}{prec:<10.3f}{rec:<10.3f}{f1:<10.3f}')

        logger.debug('\n'.join(logs))
        return mean_res

    def _get_mean_results(self, results: list) -> Metric:
        logs = [
            'averages:',
            f'{"precision":<10}{"recall":<10}{"f1":<10}'
        ]
        mean_metric = Metric([], [])
        prec = np.array([m.precision() for m in results if m is not None]).mean()
        rec = np.array([m.recall() for m in results if m is not None]).mean()
        fpr = np.array([m.fpr() for m in results if m is not None]).mean()
        f1 = np.array([m.f1() for m in results if m is not None]).mean()
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
            logs.append(
                format_cell(test_set[i]) + format_cell(results[i].precision()) + format_cell(
                    results[i].recall()) + format_cell(results[i].f1()))

        logger.info('\n'.join(logs))

    def __save_charts(self, best_thr: float, results: dict, thresholds: list, initial_depth: int, max_depth: int):
        precs_list = [results[thr].precision() for thr in thresholds]
        recs_list = [results[thr].recall() for thr in thresholds]
        f1s_list = [results[thr].f1() for thr in thresholds]
        fprs_list = [results[thr].fpr() for thr in thresholds]
        best_res = results[best_thr]

        pyplot.figure()
        pyplot.subplot(221)
        pyplot.plot(thresholds, precs_list)
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.scatter([best_thr], [best_res.precision()], c='r', marker='o')
        pyplot.title('precision')
        pyplot.subplot(222)
        pyplot.plot(thresholds, recs_list)
        pyplot.scatter([best_thr], [best_res.recall()], c='r', marker='o')
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.title('recall')
        pyplot.subplot(223)
        pyplot.plot(thresholds, f1s_list)
        pyplot.scatter([best_thr], [best_res.f1()], c='r', marker='o')
        pyplot.axis([0, pyplot.axis()[1], 0, 1])
        pyplot.title('F1')
        pyplot.subplot(224)
        pyplot.plot(fprs_list, recs_list)
        pyplot.scatter([best_res.fpr()], [best_res.recall()], c='r', marker='o')
        pyplot.title('ROC curve')
        pyplot.axis([0, 1, 0, 1])
        results_path = os.path.join(settings.BASE_PATH, 'results')
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        filename = os.path.join(results_path,
                                f'{self.project.name}-{self.method}-{initial_depth}-{max_depth}.png')
        pyplot.savefig(filename)
        # pyplot.show()

    def __sequences_to_obs_states(self, sequences, memm):
        observations = []
        states = []
        for seq in sequences:
            for obs, state in seq:
                observations.append(obs)
                states.append(state)
        observations = np.array(observations)[:, memm.orig_indexes]
        states = np.array(states)
        return observations, states

    @abc.abstractmethod
    def _get_validate_n_jobs(self):
        pass

    @abc.abstractmethod
    def _evid_multiprocessed(self):
        pass


class DefaultTester(ProjectTester):
    def train(self, **kwargs):
        return self._do_train(multi_processed=False)

    def _do_test(self, test_set, thresholds, graph, trees, initial_depth=0, max_depth=None, model=None):
        return test_cascades(test_set, self.method, model, thresholds, initial_depth, max_depth, self.criterion, trees,
                             graph)

    def _get_validate_n_jobs(self):
        return None

    def _evid_multiprocessed(self):
        return False


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

        test_res = {thr: [] for thr in thresholds} if isinstance(thresholds, list) else []

        # Collect results of the processes.
        for res in results:
            cur_res = res.get()
            if isinstance(thresholds, list):
                for thr in thresholds:
                    test_res[thr].extend(cur_res[thr])
            else:
                test_res.extend(cur_res)

        return test_res

    def _get_validate_n_jobs(self):
        return settings.PROCESS_COUNT

    def _evid_multiprocessed(self):
        return True
