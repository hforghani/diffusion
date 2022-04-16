import argparse
# from profilehooks import timecall, profile
import pprint
from numbers import Number

import numpy as np

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


def run_predict(method_name: str, project_name: str, validation: bool, criterion: Criterion, initial_depth: int,
                max_depth: int, multi_processed: bool, eco: bool, param: list) -> Metric:
    project = Project(project_name)
    method = Method(method_name)
    param = extract_param(param, validation)
    threshold = param['threshold']
    del param['threshold']

    # Log the test configuration.
    logger.info(f'{"db":<20}| {project.db}')
    logger.info(f'{"project":<20}| {project_name}')
    logger.info(f'{"method":<20}| {method_name}')
    logger.info(f'{"initial depth":<20}| {initial_depth}')
    logger.info(f'{"max depth":<20}| {max_depth}')
    logger.info(f'{"threshold":<20}| {threshold}')
    for key, value in param.items():
        logger.info(f'{key:<20}| {value}')

    if multi_processed:
        tester = MultiProcTester(project, method, criterion, eco)
    else:
        tester = DefaultTester(project, method, criterion, eco)

    if validation:
        return tester.run_validation_test(threshold, initial_depth, max_depth, **param)
    else:
        return tester.run_test(threshold, initial_depth, max_depth, **param)


def extract_param(param, validation):
    new_param = {}
    for items in param:
        if len(items) == 4:
            param_name, start, end, step = items
            start, end, step = float(start), float(end), float(step)
            new_param[param_name] = np.arange(start, end, step).tolist()
            new_param[param_name].append(end)
        elif len(items) == 2:
            try:
                new_param[items[0]] = int(items[1])
            except ValueError:
                try:
                    new_param[items[0]] = float(items[1])
                except ValueError:
                    new_param[items[0]] = items[1]
        else:
            raise ValueError('invalid format for params option')
    if not validation and any(isinstance(value, list) for value in new_param.values()):
        raise ValueError('2 arguments must be given to option "param" in test mode')
    if validation and all(isinstance(value, float) for value in new_param.values()):
        raise ValueError('At least one --param option with 4 arguments must be given in validation mode')

    return new_param


@time_measure('info')
def handle(args):
    res = run_predict(args.method, args.project, args.validation, Criterion(args.criterion), args.initial_depth,
                      args.max_depth, args.multi_processed, args.eco, args.param)

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
    parser.add_argument("-v", "--validation", action='store_true', default=False,
                        help="learn the best threshold in validation stage")
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
    parser.add_argument("--param", nargs="+", action='append',
                        help="additional parameters given to the method. In the validation mode use in the format "
                             "[--param <param_name> <from_value> <to_value> <step>] and in the test mode use in the "
                             "format [--param <param_name> <value>]")

    args = parser.parse_args()

    if all(item[0] != 'threshold' for item in args.param):
        parser.error('parameter "threshold" must be given by param option')

    handle(args)


if __name__ == '__main__':
    main()
