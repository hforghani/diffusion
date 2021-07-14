import argparse
import logging
import random
import time
import traceback

import settings
from cascade.models import Project
from db.managers import DBManager
import numpy as np

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('displaytree')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Samples a subset of cascades and separate training, validation and test sets and save them into the project data'

    def add_arguments(self, parser):
        parser.add_argument('--min', type=int, default=0, help='min number of cascade users')
        parser.add_argument('--max', type=int, help='max number of cascade users')
        parser.add_argument("-n", "--number", type=int,
                            help="number of samples consisting training, validation and test sets")
        parser.add_argument("-d", "--mindepth", type=int, dest="min_depth", default=0,
                            help="minimum depth of cascade trees of memes")
        parser.add_argument("-p", "--project", type=str, help="project name")

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, args):
        try:
            start = time.time()
            query = {}
            if args.min:
                query['size'] = {'$gte': args.min}
            if args.max:
                query.setdefault('size', {})
                query['size'].update({'$lte': args.max})
            if args.min_depth:
                query['depth'] = {'$gte': args.min_depth}

            logger.debug('query: %s', query)
            db = DBManager().db
            memes = list(db.memes.find(query, ['_id']))
            memes = [m['_id'] for m in memes]

            if len(memes) < args.number:
                raise ValueError(
                    f'Number of cascades between min and max size ({len(memes)}) is less than given sample number')

            samples = list(np.random.choice(memes, args.number, replace=False))
            samples = [str(_id) for _id in samples]
            random.shuffle(samples)

            val_ratio, test_ratio = 0.15, 0.15
            val_num = round(val_ratio * args.number)
            test_num = round(test_ratio * args.number)
            val_set = samples[:val_num]
            test_set = samples[val_num:val_num + test_num]
            train_set = samples[val_num + test_num:]
            project = Project(args.project)
            project.save_sets(train_set, val_set, test_set)

            logger.info('command done in %f min' % ((time.time() - start) / 60))

        except:
            logger.error(traceback.format_exc())
            raise


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
