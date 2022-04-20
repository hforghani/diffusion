import argparse
import itertools

import numpy as np

from cascade.metric import Metric
from cascade.models import Project
from cascade.testers import MultiProcTester, DefaultTester
from diffusion.enum import Method, Criterion
from settings import logger
from utils.time_utils import time_measure


def run_predict(method_name: str, project_name: str, criterion: Criterion, initial_depth: int,
                max_depth: int, multi_processed: bool, eco: bool, params: list):
    project = Project(project_name)
    method = Method(method_name)
    params = extract_param(params)
    threshold = params['threshold']
    del params['threshold']

    # Log the test configuration.
    logger.info(f'{"db":<20}| {project.db}')
    logger.info(f'{"project":<20}| {project_name}')
    logger.info(f'{"method":<20}| {method_name}')
    logger.info(f'{"initial depth":<20}| {initial_depth}')
    logger.info(f'{"max depth":<20}| {max_depth}')
    logger.info(f'{"threshold":<20}| {threshold}')
    for key, value in params.items():
        logger.info(f'{key:<20}| {value}')

    tunables = {key: params[key] for key in params if key != 'iterations' and isinstance(params[key], list)}
    nontunables = {key: params[key] for key in params if key not in tunables}
    _, val_set, test_set = project.load_sets()

    tunable_params = list(tunables.keys())
    combinations = itertools.product(*[tunables[param] for param in tunable_params])
    best_params, best_f1 = None, -1

    for comb in combinations:
        cur_param = dict(zip(tunable_params, comb))
        cur_param.update(nontunables)
        logger.info('{0} running with params {1} {0}'.format('=' * 20, cur_param))

        if multi_processed:
            tester = MultiProcTester(project, method, criterion, eco)
        else:
            tester = DefaultTester(project, method, criterion, eco)
        mean_res, res = tester.run_validation_test(threshold, initial_depth, max_depth, **cur_param)
        if mean_res.f1() > best_f1:
            best_params, best_f1 = cur_param, mean_res.f1()

    return best_params, best_f1


def extract_param(param):
    new_param = {}
    for items in param:
        if len(items) == 4:
            param_name, start, end, step = items
            start, end, step = float(start), float(end), float(step)
            values = [round(val, 5) for val in np.arange(start, end, step)]
            if end not in values:
                values.append(end)
            new_param[param_name] = values
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
    if all(isinstance(value, float) for value in new_param.values()):
        raise ValueError('At least one --param option with 4 arguments must be given in validation mode')

    return new_param


@time_measure('info')
def handle(args):
    best_params, best_f1 = run_predict(args.method, args.project, Criterion(args.criterion),
                                       args.initial_depth, args.max_depth, args.multi_processed, args.eco, args.param)

    logger.info('best F1 = %f with parameters %s', best_f1, best_params)


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
