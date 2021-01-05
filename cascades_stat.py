import argparse
import json
import logging
import time
import traceback

import settings
from memm.db import DBManager
from matplotlib import pyplot
import numpy as np

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('displaytree')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Show statistics of the cascades between a min and max user count'

    def add_arguments(self, parser):
        parser.add_argument(
            '--min',
            type=int,
            default=0,
            help='min user count',
        )
        parser.add_argument(
            '--max',
            type=int,
            help='max user count',
        )
        parser.add_argument(
            '--out',
            type=str,
            help='file path in which we want to output the list of meme ids',
        )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, args):
        try:
            start = time.time()
            memes = list(DBManager().db.memes.find({}, ['_id', 'count']).sort('count', -1))
            mcounts = np.array([m['count'] for m in memes])
            min_count, max_count = min(mcounts), max(mcounts)
            print(f'min of all: {min_count}')
            print(f'max of all: {max_count}')
            range_max = args.max if args.max is not None else max_count
            bins_num = 100
            counts, bins = np.histogram(mcounts, bins=bins_num, range=(args.min, range_max))
            for i in range(len(counts)):
                print(f'{bins[i]} - {bins[i + 1]} : {counts[i]}')
            print(f'count between {args.min} and {max_count}: {sum(counts)}')

            if args.out:
                self.output_memeids(args.out, memes, args.min, range_max)

            pyplot.bar(bins[:-1], counts, width=(range_max - args.min) / bins_num)
            pyplot.show()
            logger.info('command done in %f min' % ((time.time() - start) / 60))
        except:
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    def output_memeids(filename, memes, min_count, max_count):
        meme_ids = [str(meme['_id']) for meme in memes if min_count <= meme['count'] <= max_count]
        with open(filename, 'w') as f:
            json.dump(meme_ids, f)


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
