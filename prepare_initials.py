import argparse

from cascade.models import Project
from settings import logger


def main():
    parser = argparse.ArgumentParser(
        "Prepares initial data for the project given: graph, activation sequences, and trees")
    parser.add_argument('project', type=str, help='project name')
    args = parser.parse_args()

    project = Project(args.project)
    project.load_or_extract_graph_seq()
    project.load_trees()
    logger.info('done')


if __name__ == '__main__':
    main()
