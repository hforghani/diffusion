import argparse
# from profilehooks import timecall, profile
from numbers import Number

from cascade.metric import Metric
from cascade.models import Project
import settings
from cascade.testers import MultiProcTester, DefaultTester
from diffusion.enum import Method, Criterion
from settings import logger
from utils.time_utils import time_measure


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)


def get_def_thr(method):
    if method == Method.MLN_PRAC:
        return 5, 10
    elif method == Method.MLN_ALCH:
        return 0, 0.5
    else:
        return 0, 1


def run_predict(method_name: str, project_name: str, validation: bool, criterion: Criterion, min_threshold: Number,
                max_threshold: Number, thresholds_count: int, initial_depth: int, max_depth: int, iterations: int,
                multi_processed: bool, eco: bool) -> Metric:
    project = Project(project_name)
    method = Method(method_name)

    if validation:
        default_thr = get_def_thr(method)
        thr_min = min_threshold if min_threshold else default_thr[0]
        thr_max = max_threshold if max_threshold else default_thr[1]
        if thr_max < thr_min:
            raise ValueError('the min threshold is greater than the max threshold')
        step = (thr_max - thr_min) / (thresholds_count - 1)
        thresholds = [step * i + thr_min for i in range(thresholds_count)]
    else:
        thresholds = [min_threshold]

    # Log the test configuration.
    logger.info('{0} DB : {1} {0}'.format('=' * 20, project.db))
    logger.info('{0} PROJECT : {1} {0}'.format('=' * 20, project_name))
    logger.info('{0} METHOD : {1} {0}'.format('=' * 20, method_name))
    logger.info('{0} INITIAL DEPTH : {1} {0}'.format('=' * 20, initial_depth))
    logger.info('{0} MAX DEPTH : {1} {0}'.format('=' * 20, max_depth))
    logger.info('{0} TESTING ON THRESHOLD(S) : {1} {0}'.format('=' * 20, thresholds))

    if multi_processed:
        tester = MultiProcTester(project, method, criterion, eco)
    else:
        tester = DefaultTester(project, method, criterion, eco)

    if validation:
        return tester.run_validation_test(thresholds, initial_depth, max_depth, iterations=iterations)
    else:
        if min_threshold is None:
            raise ValueError('must specify the threshold via -t option when --validation option is not given')
        return tester.run_test(min_threshold, initial_depth, max_depth, iterations=iterations)


@time_measure('info')
def handle(args):
    res = run_predict(args.method, args.project, args.validation, Criterion(args.criterion), args.min_threshold,
                      args.max_threshold, args.thresholds_count, args.initial_depth, args.max_depth, args.iterations,
                      args.multi_processed, args.eco)

    if res is not None:
        logger.info('final precision = %.3f, recall = %.3f, f1 = %.3f, fpr = %.3f', res.precision(), res.recall(),
                    res.f1(), res.fpr())


def main():
    parser = argparse.ArgumentParser('Test information diffusion prediction')
    parser.add_argument("-p", "--project", type=str, dest="project", help="project name")
    parser.add_argument("-m", "--method", type=str, dest="method", required=True, choices=[e.value for e in Method],
                        help="the method by which we want to test")
    parser.add_argument("-C", "--criterion", choices=[e.value for e in Criterion], default="nodes",
                        help="the criterion on which the evaluation is done")
    parser.add_argument("-t", "--min_threshold", type=float,
                        help="minimum threshold to apply on the method")
    parser.add_argument("-T", "--max_threshold", type=float,
                        help="maximum threshold to apply on the method")
    parser.add_argument("-v", "--validation", action='store_true', default=False,
                        help="learn the best threshold in validation stage")
    parser.add_argument("-c", "--thresh-count", type=int, dest="thresholds_count", default=10,
                        help="in the case the argument --validation is given, this argument specifies the number "
                             "of thresholds to test between min and max thresholds in validation stage")
    parser.add_argument("-i", "--init-depth", type=int, dest="initial_depth", default=0,
                        help="the maximum depth of the initial nodes")
    parser.add_argument("-d", "--max-depth", type=int, dest="max_depth",
                        help="the maximum depth of cascade prediction")
    parser.add_argument("-M", "--multiprocessed", action='store_true', dest="multi_processed", default=False,
                        help="If this option is given, the task is ran on multiple processes")
    parser.add_argument("-e", "--eco", action='store_true', default=False,
                        help="If this option is given, the prediction is done in economical mode e.t. Memory consumption "
                             "is decreased and data is stored in DB and loaded everytime needed instead of storing in "
                             "RAM. Otherwise, no data is stored in DB.")
    parser.add_argument("--iterations", type=int, help="the maximum learning iterations")

    args = parser.parse_args()
    handle(args)


if __name__ == '__main__':
    main()
