# -*- coding: utf-8 -*-
from optparse import make_option
import traceback
from django.core.management.base import BaseCommand
import time
from cascade.models import Project
from cascade.saito import Saito


class Command(BaseCommand):
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

    def handle(self, *args, **options):
        try:
            # Get project or raise exception.
            project_name = options['project']
            if project_name is None:
                raise Exception('project not specified')
            project = Project(project_name)

            # Calculate Saito parameters.
            start = time.time()
            Saito(project).calc_parameters(iterations=options['iterations'])
            self.stdout.write('command done in %.2f min' % ((time.time() - start) / 60.0))
        except:
            self.stdout.write(traceback.format_exc())
            raise
