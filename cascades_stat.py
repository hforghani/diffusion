import argparse
import json
import logging
import os
import time
import traceback

import settings
from db.managers import DBManager
from matplotlib import pyplot
import numpy as np

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('displaytree')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = 'Show statistics of the cascades between a min and max user count'

    def add_arguments(self, parser):
        parser.add_argument('--min', type=int, default=0, help='min user count')
        parser.add_argument('--max', type=int, help='max user count')
        parser.add_argument('--idout', type=str, help='output file path for list of meme ids')
        parser.add_argument('--pltout', type=str, required=True, help='output file path for plot image')

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, args):
        try:
            start = time.time()
            if args.max is None and args.min is None:
                query = {'size': {'$ne': None}}
            else:
                query = {'size': {}}
                if args.max:
                    query['size']['$lte'] = args.max
                if args.min:
                    query['size']['$gte'] = args.min

            memes = DBManager().db.memes.find(query, {'_id': int(args.idout is not None), 'size': 1})
            msizes = np.array(sorted([m['size'] for m in memes], reverse=True))
            min_size, max_size = min(msizes), max(msizes)
            print(f'min of all: {min_size}')
            print(f'max of all: {max_size}')
            range_max = args.max if args.max is not None else max_size
            bins_num = 100
            counts, bins = np.histogram(msizes, bins=bins_num, range=(args.min, range_max))
            for i in range(len(counts)):
                print(f'{bins[i]} - {bins[i + 1]} : {counts[i]}')
            print(f'count between {args.min} and {max_size}: {sum(counts)}')

            if args.idout:
                self.output_memeids(args.idout, memes, args.min, range_max)

            pyplot.bar(bins[:-1], counts, width=(range_max - args.min) / bins_num)
            pyplot.savefig(args.pltout)
            # pyplot.show()
            logger.info('command done in %f min' % ((time.time() - start) / 60))
        except:
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    def output_memeids(filename, memes, min_size, max_size):
        meme_ids = [str(meme['_id']) for meme in memes if min_size <= meme['count'] <= max_size]
        with open(filename, 'w') as f:
            json.dump(meme_ids, f)


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
