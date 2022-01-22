# -*- coding: utf-8 -*-
import argparse
import logging
from cascade.models import Project
from cascade.aslt import AsLT
import settings
from utils.time_utils import time_measure

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('calc_aslt')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Calculates diffusion model parameters using AsLT method'

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

    @time_measure()
    def handle(self, args):
        # Get project or raise exception.
        project_name = args.project
        if project_name is None:
            raise Exception('project not specified')
        project = Project(project_name)

        # Calculate AsLT parameters.
        aslt = AsLT(project)
        train_set, _, _ = aslt.project.load_sets()
        aslt.project.delete_param(aslt.w_param_name)
        aslt.project.delete_param(aslt.r_param_name)
        aslt.calc_parameters(train_set, args.iterations, multi_processed=True, eco=True)


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
