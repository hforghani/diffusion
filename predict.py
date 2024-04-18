import argparse
# from profilehooks import timecall, profile
from cascade.metric import Metric, METRICS
from cascade.models import Project
from cascade.testers import MultiProcTester, DefaultTester
from config.predict_config import PredictConfig, set_config
from diffusion.enum import Method, Criterion
from settings import logger
from utils.time_utils import time_measure


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)


def run_predict() -> Metric:
    config = PredictConfig()
    project = Project(config.project)
    method = Method(config.method)

    # Log the test configuration.
    logger.info(f'{"db":<20}| {project.db}')
    logger.info(f'{"project":<20}| {config.project}')
    logger.info(f'{"method":<20}| {config.method}')
    logger.info(f'{"initial depth":<20}| {config.init_depth}')
    logger.info(f'{"max depth":<20}| {config.max_depth}')
    logger.info(f'{"criterion":<20}| {config.criterion.value}')
    for key, value in config.params.items():
        logger.info(f'{key:<20}| {value}')

    if config.multiprocessed:
        tester = MultiProcTester(project, method, config.criterion, config.eco)
    else:
        tester = DefaultTester(project, method, config.criterion, config.eco)

    mean_res, res, _ = tester.run(config.init_depth, config.max_depth, n_iter=config.n_iter, **config.params)

    return mean_res


@time_measure('info')
def handle():
    res = run_predict()

    if res is not None:
        logger.info(f"final {', '.join(f'{metric}: {value:.3f}' for metric, value in res.metrics.items())}")


def main():
    parser = argparse.ArgumentParser('Test information diffusion prediction')
    parser.add_argument("-p", "--project", type=str, dest="project", help="project name")
    parser.add_argument("-m", "--method", type=str, dest="method", required=True, choices=[e.value for e in Method],
                        help="the method by which we want to test")
    parser.add_argument("-C", "--criterion", choices=[e.value for e in Criterion], default="nodes",
                        help="the criterion on which the evaluation is done")
    parser.add_argument("-i", "--init-depth", type=int, dest="initial_depth", default=0,
                        help="the maximum depth of the initial nodes")
    parser.add_argument("-d", "--max-depth", type=int, dest="max_depth",
                        help="the maximum depth of cascade prediction")
    parser.add_argument("-M", "--multiprocessed", action='store_true', dest="multi_processed", default=False,
                        help="If this option is given, the task is ran on multiple processes")
    parser.add_argument("-e", "--eco", action='store_true', default=False,
                        help="If this option is given, the already saved trained models is fetched from db and used. "
                             "Also it will be saved if has not been saved.")
    parser.add_argument("--param", nargs="+", action='append',
                        help="additional parameters given to the method. If the format [--param <param_name> "
                             "<from_value> <to_value>] is used at least once, randomized search cross-validation with "
                             "3 folds is done. If the only parameters with this format is 'threshold', grid search "
                             "cross-validation with 3 folds id executed. If the format --param <param_name> <value>] "
                             "is used for all parameters, just one run is done using the specified parameters.")
    parser.add_argument("-n", "--n-iter", type=int, dest='n_iter', default=100,
                        help="Number of randomized search iterations. Used when --param is given with a range of values.")
    parser.add_argument("-a", "--additional", choices=METRICS, dest="additional_metrics", nargs="+",
                        help="additional reported metrics")

    args = parser.parse_args()

    set_config(args.method, args.project, args.criterion, args.initial_depth, args.max_depth,
               args.multi_processed, args.eco, args.param, args.n_iter, args.additional_metrics)

    if all(item[0] != 'threshold' for item in args.param):
        parser.error('parameter "threshold" must be given by param option')

    handle()


if __name__ == '__main__':
    main()
