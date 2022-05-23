import argparse

from cascade.models import Project
from cascade.testers import DefaultTester
from settings import logger


def main():
    parser = argparse.ArgumentParser(
        "Prepares initial data for the project given: graph, activation sequences, and trees")
    parser.add_argument('project', type=str, help='project name')
    args = parser.parse_args()

    project = Project(args.project)
    logger.info('extracting training graph and act sequences if not exist ...')
    project.load_or_extract_graph_seq()
    folds = DefaultTester(project, None).get_cross_val_folds(3)
    logger.info('extracting graphs of 3 folds if not exist ...')
    for fold in folds:
        project.load_or_extract_graph(fold)
    logger.info('extracting trees if not exist ...')
    project.load_trees()
    logger.info('done')


if __name__ == '__main__':
    main()
