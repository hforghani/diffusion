# -*- coding: utf-8 -*-
import argparse
# from profilehooks import timecall, profile
from argparse import ArgumentError

from cascade.models import Project
import settings
from cascade.testers import MultiProcTester, DefaultTester
from settings import logger
from utils.time_utils import time_measure


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)


class Command:
    help = 'Test information diffusion prediction'

    def add_arguments(self, parser):
        parser.add_argument("-p", "--project", type=str, dest="project", help="project name")
        parser.add_argument("-m", "--method", type=str, dest="method", required=True,
                            choices=['mlnprac', 'mlnalch', 'binmemm', 'floatmemm', 'aslt', 'avg'],
                            help="the method by which we want to test")
        parser.add_argument("-t", "--min_threshold", type=float,
                            help="minimum threshold to apply on the method")
        parser.add_argument("-T", "--max_threshold", type=float,
                            help="maximum threshold to apply on the method")
        parser.add_argument("-v", "--validation", action='store_true', default=False,
                            help="learn the best threshold in validation stage")
        parser.add_argument("-c", "--thresh-count", type=int, dest="thresholds_count", default=10,
                            help="in the case the argument --validation is given, this argument specifies the number "
                                 "of thresholds to test between min and max thresholds in validation stage")
        parser.add_argument("-i", "--init-depth", type=int, dest="initial_depth", default=0,
                            help="the maximum depth of the initial nodes")
        parser.add_argument("-d", "--max-depth", type=int, dest="max_depth",
                            help="the maximum depth of cascade prediction")
        parser.add_argument("-u", "--multiprocessed", action='store_true', dest="multi_processed", default=False,
                            help="if this options is given, the task is ran on multiple processes")

    @time_measure('info')
    def handle(self, args):
        method = args.method
        project_name = args.project
        project = Project(project_name)

        if args.validation:
            thres_min = args.min_threshold if args.min_threshold else settings.THRESHOLDS[method][0]
            thres_max = args.max_threshold if args.max_threshold else settings.THRESHOLDS[method][1]
            if thres_max < thres_min:
                raise ArgumentError('the min threshold is greater than the max threshold')
            step = (thres_max - thres_min) / (args.thresholds_count - 1)
            thresholds = [step * i + thres_min for i in range(args.thresholds_count)]
        else:
            thresholds = [args.min_threshold]

        # Log the test configuration.
        logger.info('{0} DB : {1} {0}'.format('=' * 20, project.db))
        logger.info('{0} PROJECT : {1} {0}'.format('=' * 20, project_name))
        logger.info('{0} METHOD : {1} {0}'.format('=' * 20, method))
        logger.info('{0} INITIAL DEPTH : {1} {0}'.format('=' * 20, args.initial_depth))
        logger.info('{0} MAX DEPTH : {1} {0}'.format('=' * 20, args.max_depth))
        logger.info('{0} TESTING ON THRESHOLD(S) : {1} {0}'.format('=' * 20, thresholds))

        if args.multi_processed:
            tester = MultiProcTester(project, method)
        else:
            tester = DefaultTester(project, method)

        if args.validation:
            prec, rec, f1, fpr = tester.run_validation_test(thresholds, args.initial_depth, args.max_depth)
        else:
            if args.min_threshold is None:
                raise ArgumentError('must specify the threshold via -t option when --validation options is not given')
            prec, rec, f1, fpr = tester.run_test(args.min_threshold, args.initial_depth, args.max_depth)

        if prec is not None:
            logger.info('final precision = %.3f, recall = %.3f, f1 = %.3f, fpr = %.3f', prec, rec, f1, fpr)


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
