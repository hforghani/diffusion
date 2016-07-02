# -*- coding: utf-8 -*-
from optparse import make_option
import traceback
from django.core.management.base import BaseCommand
import time
from cascade.saito import Saito


class Command(BaseCommand):
    help = 'Calculates diffusion model parameters using Saito method'

    option_list = BaseCommand.option_list + (
        make_option(
            "-i",
            "--iterations",
            type="int",
            default=3,
            dest="iterations",
            help="number of iterations"
        ),
    )

    def handle(self, *args, **options):
        try:
            start = time.time()
            Saito().calc_parameters(iterations=options['iterations'])
            self.stdout.write('command done in %.2f min' % ((time.time() - start) / 60.0))
        except:
            self.stdout.write(traceback.format_exc())
            raise
