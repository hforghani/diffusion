# -*- coding: utf-8 -*-
import traceback

from django.core.management.base import BaseCommand
import time

from cascade.models import Project, CascadeTree


class Command(BaseCommand):
    help = 'Display cascade tree of a meme from a project'

    def add_arguments(self, parser):
        parser.add_argument(
            'meme_id',
            type=int,
            help='meme id',
        )
        parser.add_argument(
            '-p',
            '--project',
            type=str,
            help='project name',
        )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()
            meme_id = options['meme_id']
            if options['project']:
                project = Project(options['project'])
                trees = project.load_trees()
                tree = trees[meme_id]
            else:
                tree = CascadeTree().extract_cascade(meme_id)

            self.stdout.write('\n' + tree.render())
            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
