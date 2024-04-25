import argparse
from functools import reduce

from cascade.models import Project
from cascade.testers import DefaultTester
from settings import logger


def main():
    parser = argparse.ArgumentParser(
        "Prepares initial data for the project given: graph, activation sequences, and trees")
    parser.add_argument('project', type=str, help='project name')
    args = parser.parse_args()

    project = Project(args.project)

    logger.info('extracting trees if not exist ...')
    project.load_trees()

    logger.info('extracting training graph and act sequences if not exist ...')
    project.load_or_extract_graph_seq()
    folds_num = 3
    folds = DefaultTester(project, None).get_cross_val_folds(folds_num)
    logger.info('extracting graphs of %d folds if not exist ...', folds_num)

    for i in range(folds_num):
        logger.info('extracting graph missing fold %d ...', i + 1)
        train_set = reduce(lambda x, y: x + y, folds[:i] + folds[i + 1:], [])
        project.load_or_extract_graph(train_set)

    logger.info('done')


if __name__ == '__main__':
    main()
