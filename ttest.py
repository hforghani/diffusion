import argparse

from scipy import stats
import numpy as np

import settings
from diffusion.enum import Method, Criterion
from cascade.models import Project
from cascade.testers import DefaultTester, MultiProcTester
from settings import logger
from utils.time_utils import time_measure


def get_params(project_name, method):
    if method in settings.PARAM:
        params = settings.PARAM[method]
        if project_name in params:
            return params[project_name]
    return {}


def multiple_run(methods: list, project_name: str, multi_processed: bool, criterion: Criterion) -> dict:
    project = Project(project_name)
    eco = True

    if multi_processed:
        testers = {method: MultiProcTester(project, method, criterion, eco=eco) for method in methods}
    else:
        testers = {method: DefaultTester(project, method, criterion, eco=eco) for method in methods}

    thresholds = [i / 100 for i in range(101)]
    results = {}  # dict of methods to lists of f1 values.
    mean_results = {}  # dict of methods to mean f1 values.

    for method in methods:
        logger.info('running prediction using method %s ...', method.value)
        params = get_params(project_name, method)
        logger.info('params = %s', params)
        mean_res, res = testers[method].run_validation_test(thresholds, 0, None, **params)
        results[method] = np.array([metric.f1() for metric in res])
        mean_results[method] = mean_res.f1()

    report_results(mean_results, results, methods)

    return results


def report_results(mean_results, results, methods):
    logger.info('Mean results:\n' +
                ' ' * 10 + ''.join(f'{method.value:<15}' for method in methods) +
                '\nf1 =      ' + ''.join(f'{mean_results[method]:<15.3}' for method in methods))
    res0 = results[methods[0]]
    logs = [
        'T-test results:',
        f'{"method1":<20}{"method2":<20}{"t-value":<20}{"1-tail p-value":<20}'
    ]
    for method in methods[1:]:
        stat = stats.ttest_ind(res0, results[method])
        logs.append(f'{methods[0].value:<20}{method.value:<20}{stat.statistic:<20.3}{stat.pvalue / 2:<20.3}')
    logger.info('\n'.join(logs))


@time_measure()
def main():
    parser = argparse.ArgumentParser('Test information diffusion prediction')
    parser.add_argument("-p", "--project", help="project name")
    parser.add_argument("-m", "--methods", nargs="+", required=True, choices=[e.value for e in Method],
                        help="the methods by which we want to test")
    parser.add_argument("-M", "--multiprocessed", action='store_true', dest="multi_processed", default=False,
                        help="if this option is given, the task is ran on multiple processes")
    parser.add_argument("-C", "--criterion", choices=[e.value for e in Criterion], default="nodes",
                        help="the criterion on which the evaluation is done")
    args = parser.parse_args()

    methods = [Method(met) for met in args.methods]
    if len(methods) < 2:
        parser.error('At least 2 methods must be given to do t-test')

    multiple_run(methods, args.project, args.multi_processed, Criterion(args.criterion))


if __name__ == '__main__':
    main()
