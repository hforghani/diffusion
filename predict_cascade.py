import argparse
from bson import ObjectId

from cascade.models import Project
import settings
from cascade.testers import DefaultTester
from settings import logger
from utils.time_utils import time_measure


def add_arguments(parser):
    parser.add_argument("-p", "--project", type=str, dest="project", help="project name", required=True)
    parser.add_argument("-c", "--cascade", type=str, dest="cascade_id", help="cascade id to predict", required=True)
    parser.add_argument("-m", "--method", type=str, dest="method", required=True,
                        choices=['mlnprac', 'mlnalch', 'memm', 'aslt', 'avg'],
                        help="the method by which we want to test")
    parser.add_argument("-t", "--threshold", type=float, help="the threshold to apply on the method")
    parser.add_argument("--thresh-count", type=int, dest="thresholds_count", default=10,
                        help="in the case the argument --validation is given, this argument specifies the number "
                             "of thresholds to test between min and max thresholds specified in local_settings.py "
                             "in validation stage")
    parser.add_argument("-i", "--init-depth", type=int, dest="initial_depth", default=0,
                        help="the maximum depth for the initial nodes")
    parser.add_argument("-d", "--max-depth", type=int, dest="max_depth",
                        help="the maximum depth of cascade prediction")


@time_measure('info')
def handle(args):
    method = args.method
    project_name = args.project
    project = Project(project_name)
    cascade_id = ObjectId(args.cascade_id)

    if args.threshold:
        thresholds = [args.threshold]
    else:
        thres_min, thres_max = settings.THRESHOLDS[method]
        step = (thres_max - thres_min) / (args.thresholds_count - 1)
        thresholds = [step * i + thres_min for i in range(args.thresholds_count)]

    # Log the test configuration.
    logger.info('{0} DB : {1} {0}'.format('=' * 20, project.db))
    logger.info('{0} PROJECT : {1} {0}'.format('=' * 20, project_name))
    logger.info('{0} ON CASCADE : {1} {0}'.format('=' * 20, cascade_id))
    logger.info('{0} METHOD : {1} {0}'.format('=' * 20, method))
    logger.info('{0} INITIAL DEPTH : {1} {0}'.format('=' * 20, args.initial_depth))
    logger.info('{0} MAX DEPTH : {1} {0}'.format('=' * 20, args.max_depth))
    logger.info('{0} TESTING ON THRESHOLD(S) : {1} {0}'.format('=' * 20, thresholds))

    tester = DefaultTester(project, method)
    model = tester.train()

    if args.threshold:
        tester.test([cascade_id], model, args.threshold, args.initial_depth, args.max_depth)
    else:
        best_thr, best_f1 = tester.validate([cascade_id], model, thresholds, args.initial_depth, args.max_depth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test diffusion prediction on a single cascade')
    add_arguments(parser)
    args = parser.parse_args()
    handle(args)
