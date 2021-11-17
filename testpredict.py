# -*- coding: utf-8 -*-
import argparse
import time
# from profilehooks import timecall, profile
from cascade.models import Project
import settings
from cascade.testers import MultiProcTester, DefaultTester
from settings import logger

import pydevd_pycharm

pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)


class Command:
    help = 'Test information diffusion prediction'

    def add_arguments(self, parser):
        parser.add_argument("-p", "--project", type=str, dest="project", help="project name")
        parser.add_argument("-m", "--method", type=str, dest="method", required=True,
                            choices=['mlnprac', 'mlnalch', 'memm', 'aslt', 'avg'],
                            help="the method by which we want to test")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("-t", "--threshold", type=float,
                           help="the threshold to apply on the method")
        group.add_argument("-v", "--validation", action='store_true', default=False,
                           help="learn the best threshold in validation stage")
        parser.add_argument("-c", "--thresh-count", type=int, dest="thresholds_count", default=10,
                            help="in the case the argument --validation is given, this argument specifies the number "
                                 "of thresholds to test between min and max thresholds specified in local_settings.py "
                                 "in validation stage")
        parser.add_argument("-i", "--init-depth", type=int, dest="initial_depth", default=0,
                            help="the maximum depth for the initial nodes")
        parser.add_argument("-d", "--max-depth", type=int, dest="max_depth",
                            help="the maximum depth of cascade prediction")
        parser.add_argument("-u", "--multiprocessed", action='store_true', dest="multi_processed", default=False,
                            help="run tests on multiple processes")

    def handle(self, args):
        start = time.time()
        method = args.method
        project_name = args.project

        if args.validation:
            thres_min, thres_max = settings.THRESHOLDS[method]
            step = (thres_max - thres_min) / (args.thresholds_count - 1)
            thresholds = [step * i + thres_min for i in range(args.thresholds_count)]
        else:
            thresholds = [args.threshold]

        # Log the test configuration.
        logger.info('{0} DB : {1} {0}'.format('=' * 20, settings.DB_NAME))
        logger.info('{0} PROJECT : {1} {0}'.format('=' * 20, project_name))
        logger.info('{0} METHOD : {1} {0}'.format('=' * 20, method))
        logger.info('{0} INITIAL DEPTH : {1} {0}'.format('=' * 20, args.initial_depth))
        logger.info('{0} MAX DEPTH : {1} {0}'.format('=' * 20, args.max_depth))
        logger.info('{0} TESTING ON THRESHOLD(S) : {1} {0}'.format('=' * 20, thresholds))

        project = Project(project_name)

        if args.multi_processed:
            tester = MultiProcTester(project, method)
        else:
            tester = DefaultTester(project, method)

        prec, rec, f1, fpr = tester.run(thresholds, args.initial_depth, args.max_depth)

        logger.info('final precision = %.3f, recall = %.3f, f1 = %.3f, fpr = %.3f', prec, rec, f1, fpr)

        logger.info('command done in %.2f min' % ((time.time() - start) / 60))


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
