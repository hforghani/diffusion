# -*- coding: utf-8 -*-
import argparse
import logging
import traceback

import time

from cascade.models import Project, CascadeTree
import settings


logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('displaytree')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Display cascade tree of a meme from a project'

    def add_arguments(self, parser):
        parser.add_argument(
            'meme_id',
            type=str,
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

    def handle(self, args):
        try:
            start = time.time()
            meme_id = args.meme_id
            if args.project:
                project = Project(args.project)
                trees = project.load_trees()
                tree = trees[meme_id]
            else:
                tree = CascadeTree().extract_cascade(meme_id)

            logger.info('\n' + tree.render())
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
