import argparse
import logging
import random
import time
import traceback

from cascade.models import Project
from db.managers import DBManager
import numpy as np

from settings import logger
from utils.time_utils import time_measure


class Command:
    help = 'Samples a subset of cascades and separate training, validation and test sets and save them into the project data'

    def add_arguments(self, parser):
        parser.add_argument('--min', type=int, default=0, help='min number of cascade users')
        parser.add_argument('--max', type=int, help='max number of cascade users')
        parser.add_argument("-n", "--number", type=int,
                            help="number of samples consisting training, validation and test sets")
        parser.add_argument("-D", "--mindepth", type=int, dest="min_depth", default=0,
                            help="minimum depth of cascade trees")
        parser.add_argument("-p", "--project", type=str, help="project name")
        parser.add_argument('-d', '--db', required=True, help="db name")

    def __init__(self):
        super(Command, self).__init__()

    @time_measure()
    def handle(self, args):
        self.sample_data(args.db, args.project, args.number, args.min, args.max, args.min_depth)

    def sample_data(self, db_name, project_name, samples_num=None, min_size=None, max_size=None, min_depth=None):
        try:
            query = {}
            if min_size:
                query['size'] = {'$gte': min_size}
            if max_size:
                query.setdefault('size', {})
                query['size'].update({'$lte': max_size})
            if min_depth:
                query['depth'] = {'$gte': min_depth}

            logger.debug('query: %s', query)
            db = DBManager(db_name).db
            cascades = list(db.cascades.find(query, ['_id']))
            cascades = [m['_id'] for m in cascades]

            if samples_num and len(cascades) < samples_num:
                raise ValueError(
                    f'Number of cascades between min and max size ({len(cascades)}) is less than given sample number')

            if samples_num:
                samples = list(np.random.choice(cascades, samples_num, replace=False))
            else:
                samples = cascades
            samples = [str(_id) for _id in samples]
            random.shuffle(samples)

            test_ratio = 0.3
            test_num = round(test_ratio * len(samples))
            test_set = samples[:test_num]
            train_set = samples[test_num:]
            project = Project(project_name, db=db_name)
            project.save_sets(train_set, test_set)
            logger.info('project %s sampled containing %d cascades', project_name, len(samples))
            logger.info('sizes -> training: %d, test: %d', len(samples) - test_num, test_num)

        except:
            logger.error(traceback.format_exc())
            raise


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
