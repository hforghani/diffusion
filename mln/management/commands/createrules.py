# -*- coding: utf-8 -*-
import traceback
from django.core.management.base import BaseCommand
import time
from cascade.models import Project
from mln.file_generators import PracmlnCreator, Alchemy2Creator, FileCreator


class Command(BaseCommand):
    help = ''

    CREATORS = {
        FileCreator.FORMAT_PRACMLN: PracmlnCreator,
        FileCreator.FORMAT_ALCHEMY2: Alchemy2Creator
    }

    def add_arguments(self, parser):
        parser.add_argument(
            "-p",
            "--project",
            type=str,
            dest="project",
            help="project name",
        )
        parser.add_argument(
            "-o", "--output",
            type=str,
            dest="out_file",
            help="path of output file",
        )
        parser.add_argument(
            "-f", "--format",
            type=str,
            dest="format",
            default=FileCreator.FORMAT_PRACMLN,
            help="format of files. Valid values are '{}'. The default value is '{}'.".format(
                "', '".join(self.CREATORS.keys()),
                FileCreator.FORMAT_PRACMLN
            ),
        )

    def handle(self, *args, **options):
        try:
            start = time.time()

            # Get project or raise exception.
            project_name = options['project']
            if project_name is None:
                raise Exception('project not specified')
            project = Project(project_name)

            creator_clazz = self.CREATORS[options['format']]
            creator = creator_clazz(project)
            creator.create_rules(options['out_file'])

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))

        except:
            self.stdout.write(traceback.format_exc())
            raise
