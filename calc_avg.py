# -*- coding: utf-8 -*-
import argparse
import logging
import traceback
import time

import settings
from cascade.avg import LTAvg
from cascade.models import Project

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('calc_avg')
logger.setLevel(settings.LOG_LEVEL)


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)


class Command():
    help = 'Calculates diffusion model parameters'

    def add_arguments(self, parser):
        parser.add_argument(
            "-p",
            "--project",
            type=str,
            dest="project",
            help="project name",
        )
        parser.add_argument(
            "-w",
            "--weight",
            action="store_true",
            dest="weight",
            help="just calculate diffusion weights"
        )
        parser.add_argument(
            "-d",
            "--delay",
            action="store_true",
            dest="delay",
            help="just calculate diffusion delays"
        )
        parser.add_argument(
            "-c",
            "--continue",
            action="store_true",
            dest="continu",
            help="continue from the last point"
        )

    def handle(self, args):
        try:
            start = time.time()

            # Get project or raise exception.
            project_name = args.project
            if project_name is None:
                raise Exception('project not specified')
            project = Project(project_name)

            lt_avg = LTAvg(project)
            lt_avg.calc_parameters(args.weight, args.delay, args.continu)

            logger.info('command done in %f min' % ((time.time() - start) / 60))
        except:
            logger.info(traceback.format_exc())
            raise


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
