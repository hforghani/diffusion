# -*- coding: utf-8 -*-
import argparse
import logging
import traceback
from bson import ObjectId

from cascade.models import Project, ParamTypes
import settings
from utils.time_utils import time_measure

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('displaytree')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Display cascade tree of a cascade from a project'

    def add_arguments(self, parser):
        parser.add_argument('-c', '--cascade', required=True, help='cascade id')
        parser.add_argument('-p', '--project', required=True, help='project name')

    def __init__(self):
        super(Command, self).__init__()

    @time_measure()
    def handle(self, args):
        try:
            cascade_id = args.cascade
            project = Project(args.project)
            if project.has_param('trees', ParamTypes.JSON):
                trees = project.load_trees()
                tree = trees[ObjectId(cascade_id)]
            else:
                tree = project.extract_cascade(cascade_id)

            logger.info('\n' + tree.render(digest=True))
        except:
            logger.info(traceback.format_exc())
            raise


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
