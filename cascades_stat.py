import argparse
import json
import logging
import time
import traceback

import settings
from db.managers import DBManager
from matplotlib import pyplot
import numpy as np

logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('cascades_stat')
logger.setLevel(settings.LOG_LEVEL)


def output_cascade_ids(filename, cascades):
    cascade_ids = [str(cascade['_id']) for cascade in cascades]
    with open(filename, 'w') as f:
        json.dump(cascade_ids, f)


def main(args):
    try:
        start = time.time()
        if args.max is None and args.min is None:
            query = {'size': {'$exists': True}}
        else:
            query = {'size': {}}
            if args.max:
                query['size']['$lte'] = args.max
            if args.min:
                query['size']['$gte'] = args.min

        cascades = DBManager(args.db).db.cascades.find(query, {'_id': int(args.idout is not None), 'size': 1})
        csizes = np.array([m['size'] for m in cascades])
        min_size, max_size = csizes.min(), csizes.max()
        print(f'min of all: {min_size}')
        print(f'max of all: {max_size}')
        range_max = args.max if args.max is not None else max_size
        bins_num = 100
        counts, bins = np.histogram(csizes, bins=bins_num, range=(args.min, range_max))
        for i in range(len(counts)):
            print(f'{bins[i]} - {bins[i + 1]} : {counts[i]}')
        print(f'count between {args.min} and {max_size}: {sum(counts)}')

        if args.idout:
            output_cascade_ids(args.idout, cascades)

        pyplot.bar(bins[:-1], counts, width=(range_max - args.min) / bins_num)
        pyplot.savefig(args.pltout)
        # pyplot.show()
        logger.info('command done in %f min' % ((time.time() - start) / 60))
    except:
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Show statistics of the cascades between a min and max user count')
    parser.add_argument('-d', '--db', required=True, help="db name")
    parser.add_argument('--min', type=int, default=0, help='min user count')
    parser.add_argument('--max', type=int, help='max user count')
    parser.add_argument('--idout', type=str, help='output file path for list of cascade ids')
    parser.add_argument('--pltout', type=str, required=True, help='output file path for plot image')
    args = parser.parse_args()

    main(args)
