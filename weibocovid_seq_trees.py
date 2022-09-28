import argparse

from cascade.models import Project
from settings import logger


def extract_act_seq(project_all, project):
    training_set, test_set = project.load_sets()
    sequences = project_all.load_or_extract_act_seq(training_set)
    project.save_act_sequences(sequences)


def extract_trees(project_all, project):
    training_set, test_set = project.load_sets()
    trees = project_all.load_trees()
    cascades = set(training_set + test_set)
    trees = {cid: tree for cid, tree in trees.items() if cid in cascades}
    project.save_trees(trees)


def main(project_name):
    project_all = Project('weibocovid-all')
    project = Project(project_name, 'weibocovid')
    logger.info('extracting activation sequences ...')
    extract_act_seq(project_all, project)
    logger.info('extracting trees ...')
    extract_trees(project_all, project)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Extract the activation sequence and trees of a Weibo Covid subset from the project weibocovid-all.')
    parser.add_argument("-p", "--project", required=True, help="project name")
    args = parser.parse_args()
    main(args.project)
