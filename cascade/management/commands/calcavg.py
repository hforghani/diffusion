# -*- coding: utf-8 -*-
from optparse import make_option
import traceback
from django.core.management.base import BaseCommand
import time
from cascade.avg import LTAvg
from cascade.models import Project


class Command(BaseCommand):
    help = 'Calculates diffusion model parameters'

    option_list = BaseCommand.option_list + (
        make_option(
            "-p",
            "--project",
            type="string",
            dest="project",
            help="project name",
        ),
        make_option(
            "-w",
            "--weight",
            action="store_true",
            dest="weight",
            help="just calculate diffusion weights"
        ),
        make_option(
            "-d",
            "--delay",
            action="store_true",
            dest="delay",
            help="just calculate diffusion delays"
        ),
        make_option(
            "-c",
            "--continue",
            action="store_true",
            dest="continue",
            help="continue from the last point"
        ),
    )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()

            # Get project or raise exception.
            project_name = options['project']
            if project_name is None:
                raise Exception('project not specified')
            project = Project(project_name)

            lt_avg = LTAvg(project)
            lt_avg.fit(options['weight'], options['delay'], options['continue'])

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
