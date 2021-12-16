# -*- coding: utf-8 -*-
import argparse
import logging
import traceback

import time

from cascade.models import Project, CascadeTree
import settings
from db.managers import DBManager

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('displaytree')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Display cascade sizes of a project'

    def add_arguments(self, parser):
        parser.add_argument(
            'project',
            type=str,
            help='project name',
        )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, args):
        try:
            start = time.time()
            project = Project(args.project)
            training, validation, test = project.load_sets()
            cascade_ids = training + validation + test
            db = DBManager().db
            cascades = db.memes.find({'_id': {'$in': cascade_ids}}, ['_id', 'depth', 'size'])
            print(f'{"cascade id":30}{"size":10}{"depth":10}')
            for cascade in cascades:
                print(f'{str(cascade["_id"]):30}{cascade["size"]:<10}{cascade["depth"]:<10}')
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
