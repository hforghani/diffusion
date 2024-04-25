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

            # train_set, test_set = project.load_sets()

            trees = project.load_trees()
            sizes, depths = {}, {}
            for cid, tree in trees.items():
                sizes[cid] = tree.size()
                depths[cid] = tree.depth
                # assert cid not in train_set or all(node_id in graph for node_id in tree.node_ids())
            min_size = min(sizes.values())
            max_size = max(sizes.values())
            min_depth = min(depths.values())
            max_depth = max(depths.values())

            print('Number of cascades:', len(trees))
            print('min size:', min_size)
            print('max size:', max_size)
            print('min depth:', min_depth)
            print('max depth:', max_depth)

            sorted_cids = sorted(trees, key=lambda cid: sizes[cid])
            print(f'{"cascade id":30}{"size":10}{"depth":10}')
            for cid in sorted_cids:
                print(f'{str(cid):30}{sizes[cid]:<10}{depths[cid]:<10}')
        except:
            logger.info(traceback.format_exc())
            raise


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
