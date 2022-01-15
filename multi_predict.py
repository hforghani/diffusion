import argparse

from cascade.enum import Method
from cascade.models import Project
from cascade.testers import DefaultTester
from settings import logger
from utils.time_utils import time_measure


def multiple_run(methods, depth_settings, project_name):
    project = Project(project_name)
    results = {}
    testers = {method: DefaultTester(project, method) for method in methods}
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
    parser.add_argument("-p", "--project", type=str, dest="project", help="project name")
    args = parser.parse_args()

    project = Project(args.project)
    training, validation, test = project.load_sets()
    trees = project.load_trees()
    max_test_depth = max(trees[cid].depth for cid in test)
    # methods = [Method.FLOAT_MEMM, Method.BIN_MEMM, Method.ASLT, Method.AVG]
    methods = [Method.REDUCED_FLOAT_MEMM, Method.FLOAT_MEMM, Method.REDUCED_BIN_MEMM, Method.BIN_MEMM, Method.ASLT,
               Method.AVG]

    one_step_depths = [(i, i + 1) for i in range(max_test_depth)]
    thorough_depths = [(i, None) for i in range(max_test_depth)]
    depth_settings = one_step_depths + thorough_depths
    results = multiple_run(methods, depth_settings, args.project)

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
