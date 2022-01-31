import argparse

from diffusion.enum import Method, Criterion
from cascade.models import Project
from cascade.testers import DefaultTester, MultiProcTester
from settings import logger
from utils.time_utils import time_measure


def multiple_run(methods: list, depth_settings: tuple, project_name: str, multi_processed: bool,
                 criterion: Criterion) -> dict:
    project = Project(project_name)
    results = {}

    if multi_processed:
        testers = {method: MultiProcTester(project, method, criterion, eco=True) for method in methods}
    else:
        testers = {method: DefaultTester(project, method, criterion, eco=True) for method in methods}

    for initial_depth, max_depth in depth_settings:
        cur_results = {}
        for method in methods:
            logger.info('running prediction from depth %d to %s using method %s ...', initial_depth,
                        max_depth if max_depth is not None else 'end', method.value)
            thresholds = [i / 100 for i in range(101)]
            _, _, f1, _, _ = testers[method].run_validation_test(thresholds, initial_depth, max_depth)
            cur_results[method] = f1
        results[(initial_depth, max_depth)] = cur_results
    return results


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

    project = Project(args.project)
    training, validation, test = project.load_sets()
    trees = project.load_trees()
    max_test_depth = max(trees[cid].depth for cid in test)
    methods = [Method(met) for met in args.methods]

    one_step_depths = [(i, i + 1) for i in range(max_test_depth)]
    thorough_depths = [(i, None) for i in range(max_test_depth)]
    depth_settings = one_step_depths + thorough_depths
    results = multiple_run(methods, depth_settings, args.project, args.multi_processed, Criterion(args.criterion))

    logs = [f'{"from depth":<15}{"to depth":<15}' + ''.join(f'{method.value:<15}' for method in methods)]
    for init_depth, max_depth in results:
        cur_results = results[(init_depth, max_depth)]
        row = f'{init_depth:<15}{max_depth if max_depth else "end":<15}'
        for method in cur_results:
            row += f'{cur_results[method]:<15.3}'
        logs.append(row)
    logger.info('all results:\n%s', '\n'.join(logs))


if __name__ == '__main__':
    main()
