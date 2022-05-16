# -*- coding: utf-8 -*-
import argparse
import logging
import traceback

import time

from cascade.models import Project, CascadeTree
import settings
from db.managers import DBManager
from utils.time_utils import time_measure

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('project_stat')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Display cascade sizes of a project'

    def add_arguments(self, parser):
        parser.add_argument('project', type=str, help='project name')

    def __init__(self):
        super(Command, self).__init__()

    @time_measure()
    def handle(self, args):
        try:
            project = Project(args.project)

            graph = project.load_or_extract_graph()
            print('Number of nodes:', len(graph.nodes()))

            training, test = project.load_sets()
            db = DBManager(project.db).db
            cascades = list(db.cascades.find({'_id': {'$in': training + test}}, ['_id', 'depth', 'size']))
            print('Number of cascades:', len(cascades))
            print(f'{"cascade id":30}{"size":10}{"depth":10}')
            for cascade in cascades:
                print(f'{str(cascade["_id"]):30}{cascade["size"]:<10}{cascade["depth"]:<10}')
        except:
            logger.info(traceback.format_exc())
            raise


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
