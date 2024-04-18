import argparse
import concurrent
from itertools import repeat, product
from typing import Dict

import settings
from cascade.metric import Metric
from diffusion.enum import Method, Criterion
from cascade.models import Project
from cascade.testers import DefaultTester
from settings import logger
from utils.time_utils import time_measure


def get_params(project_name, method):
    if method in settings.PARAM:
        params = settings.PARAM[method]
        if project_name in params:
            return params[project_name]
    return {}


def run(method, initial_depth, max_depth, project, eco, criterion):
    logger.info('running prediction from depth %d to %s using method %s ...', initial_depth,
                max_depth if max_depth is not None else 'end', method.value)
    tester = DefaultTester(project, method, criterion, eco=eco)
    params = get_params(project.name, method)
    logger.info('params = %s', params)
    if not params:
        logger.info('no params for %s on %s', method.value, project.name)
    mean_res, res, _ = tester.run(initial_depth, max_depth, **params)
    return mean_res.metrics


def multiple_run(methods: list, depth_settings: list, project_name: str, eco: bool,
                 criterion: Criterion) -> Dict[tuple, Dict[str, Dict[str, float]]]:
    project = Project(project_name)
    combs = list(product(depth_settings, methods))
    comb_methods = [comb[1] for comb in combs]
    init_depths = [comb[0][0] for comb in combs]
    max_depths = [comb[0][1] for comb in combs]
    logger.debug('comb_methods = %s', comb_methods)
    logger.debug('init_depths = %s', init_depths)
    logger.debug('max_depths = %s', max_depths)

    with concurrent.futures.ProcessPoolExecutor(max_workers=settings.TEST_WORKERS) as executor:
        metrics = list(executor.map(run, comb_methods, init_depths, max_depths, repeat(project), repeat(eco),
                                    repeat(criterion)))

    results = {(initial_depth, max_depth): {} for initial_depth, max_depth in depth_settings}
    for i in range(len(metrics)):
        results[(init_depths[i], max_depths[i])][comb_methods[i]] = metrics[i]

    return results


def report_results(methods, results: Dict[tuple, Dict[str, Dict[str, float]]]):
    metrics = list(next(iter(next(iter(results.values())).values())).keys())
    logs = ["All Results:\n"]
    for metric in metrics:
        logs.extend([
            f"Metric: {metric}",
            f'{"from depth":<15}{"to depth":<15}' + ''.join(f'{method.value:<15}' for method in methods),
        ])
        for init_depth, max_depth in results:
            cur_results = results[(init_depth, max_depth)]
            row = f'{init_depth:<15}{max_depth if max_depth else "end":<15}'
            for method in cur_results:
                row += f'{cur_results[method][metric]:<15.3}'
            logs.append(row)
    logger.info('\n'.join(logs))


@time_measure()
def main():
    parser = argparse.ArgumentParser('Test information diffusion prediction')
    parser.add_argument("-p", "--project", help="project name")
    parser.add_argument("-m", "--methods", nargs="+", required=True, choices=[e.value for e in Method],
                        help="the methods by which we want to test")
    parser.add_argument("-e", "--eco", action='store_true', default=False,
                        help="If this option is given, the already saved trained models is fetched from db and used. "
                             "Also it will be saved if has not been saved.")
    parser.add_argument("-C", "--criterion", choices=[e.value for e in Criterion], default="nodes",
                        help="the criterion on which the evaluation is done")
    args = parser.parse_args()

    project = Project(args.project)
    _, test = project.load_sets()
    trees = project.load_trees()

    max_test_depth = max(trees[cid].depth for cid in test)
    # The depth will not be greater than 3 since has always insufficient data and zero results.
    max_depth = min(max_test_depth, 3)

    methods = [Method(met) for met in args.methods]

    # one_step_depths = [(i, i + 1) for i in range(max_depth)]
    # thorough_depths = [(i, None) for i in range(max_depth)]
    # depth_settings = one_step_depths + thorough_depths
    depth_settings = [(0, None)]
    results = multiple_run(methods, depth_settings, args.project, args.eco, Criterion(args.criterion))

    report_results(methods, results)


if __name__ == '__main__':
    main()
