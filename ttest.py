import argparse
import concurrent.futures
from itertools import repeat

from scipy import stats
import numpy as np

import settings
from compare_results import get_params
from diffusion.enum import Method, Criterion
from cascade.models import Project
from cascade.testers import DefaultTester
from settings import logger
from utils.time_utils import time_measure


def run_method(method, project_name, eco, criterion):
    project = Project(project_name)
    tester = DefaultTester(project, method, criterion, eco=eco)
    params = get_params(project_name, method)
    logger.info('params = %s', params)
    mean_res, res, _ = tester.run(0, None, **params)
    f1_values = np.array([metric.f1() for metric in res])
    mean_f1 = mean_res.f1()
    return mean_f1, f1_values


def multiple_run(methods1: list, methods2: list, project_name: str, eco: bool,
                 criterion: Criterion) -> dict:
    methods = methods1 + methods2

    with concurrent.futures.ProcessPoolExecutor(max_workers=settings.PROCESS_COUNT) as executor:
        exec_res = executor.map(run_method, methods, repeat(project_name), repeat(eco), repeat(criterion))
    results = {}  # dict of methods to lists of f1 values.
    mean_results = {}  # dict of methods to mean f1 values.

    for method, res in zip(methods, exec_res):
        mean_f1, f1_values = res
        results[method] = f1_values
        mean_results[method] = mean_f1

    report_results(mean_results, results, methods1, methods2)

    return results


def report_results(mean_results, results, methods1, methods2):
    all_methods = methods1 + methods2
    logger.info('Mean results:\n' +
                ' ' * 10 + ''.join(f'{method.value:<15}' for method in all_methods) +
                '\nf1 =      ' + ''.join(f'{mean_results[method]:<15.3}' for method in all_methods))
    logs = [
        ' ' * 20 + ''.join(f'|{method.value:<19}{"|":<20}' for method in methods2),
        ' ' * 20 + ''.join(f'|{"t-value":<19}{"p-value":<20}' for _ in methods2)
    ]
    for m1 in methods1:
        pvalues = []
        tvalues = []
        for m2 in methods2:
            stat = stats.ttest_ind(results[m1], results[m2])
            pvalues.append(stat.pvalue / 2)
            tvalues.append(stat.statistic)
        logs.append(
            f'{m1.value:<20}' +
            ''.join(f'{tvalues[i]:<20.3}{pvalues[i]:<20.3}' for i in range(len(methods2)))
        )
    logger.info('\n'.join(logs))


@time_measure()
def main():
    parser = argparse.ArgumentParser('Test information diffusion prediction')
    parser.add_argument("-p", "--project", help="project name")
    parser.add_argument("--methods1", nargs="+", required=True, choices=[e.value for e in Method],
                        help="the methods of group 1 in student's t-test")
    parser.add_argument("--methods2", nargs="+", required=True, choices=[e.value for e in Method],
                        help="the methods of group 2 in student's t-test")
    parser.add_argument("-e", "--eco", action='store_true', default=False,
                        help="If this option is given, the already saved trained models is fetched from db and used. "
                             "Also it will be saved if has not been saved.")
    parser.add_argument("-C", "--criterion", choices=[e.value for e in Criterion], default="nodes",
                        help="the criterion on which the evaluation is done")
    args = parser.parse_args()

    methods1 = [Method(met) for met in args.methods1]
    methods2 = [Method(met) for met in args.methods2]

    multiple_run(methods1, methods2, args.project, args.eco, Criterion(args.criterion))


if __name__ == '__main__':
    main()
