# -*- coding: utf-8 -*-
import traceback

from django.core.management.base import BaseCommand
import time

from cascade.models import Project


class Command(BaseCommand):
    help = 'Display cascade tree of a meme from a project'

    def add_arguments(self, parser):
        parser.add_argument(
            'project',
            type=str,
            help='project name',
        )
        parser.add_argument(
            'meme_id',
            type=int,
            help='meme id',
        )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()
            project = Project(options['project'])
            trees = project.load_trees()
            meme_id = options['meme_id']
            tree = trees[meme_id]
            self.stdout.write('\n' + tree.render())
            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
