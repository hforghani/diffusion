# -*- coding: utf-8 -*-
import json
from optparse import make_option
import os
import traceback
from django.conf import settings
from django.db.models import Count
import numpy as np
from django.core.management.base import BaseCommand
import time
from scipy import sparse
from cascade.models import CascadeTree, Project
from crud.models import Meme, UserAccount, Reshare, Post
from mln.file_generators import PracmlnCreator, Alchemy2Creator


class Command(BaseCommand):
    help = ''

    option_list = BaseCommand.option_list + (
        make_option(
            "-p",
            "--project",
            type="string",
            dest="project",
            help="project name",
        ),
        make_option(
            "-o", "--output",
            type="string",
            dest="out_file",
            help="path of output file",
        ),
        make_option(
            "-f", "--format",
            type="string",
            dest="format",
            default="pracmln",
            help="format of files. Valid values are 'pracmln' and 'alchemy2'. The default value is 'pracmln'.",
        ),
    )

    CREATORS = {
        'pracmln': PracmlnCreator,
        'alchemy2': Alchemy2Creator
    }

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
