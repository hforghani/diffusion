# -*- coding: utf-8 -*-
import argparse
import logging
import traceback
import time
from cascade.models import Project
from cascade.saito import Saito
import settings

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('calcsaito')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Calculates diffusion model parameters using Saito method'

    def add_arguments(self, parser):
        parser.add_argument(
            "-p",
            "--project",
            type=str,
            dest="project",
            help="project name",
        )
        parser.add_argument(
            "-i",
            "--iterations",
            type=int,
            default=3,
            dest="iterations",
            help="number of iterations"
        )

    def handle(self, args):
        try:
            # Get project or raise exception.
            project_name = args.project
            if project_name is None:
                raise Exception('project not specified')
            project = Project(project_name)

            # Calculate Saito parameters.
            start = time.time()
            Saito(project).calc_parameters(iterations=args.iterations)
            logger.info('command done in %.2f min' % ((time.time() - start) / 60.0))
        except:
            logger.info(traceback.format_exc())
            raise


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
