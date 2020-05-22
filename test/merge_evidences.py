import argparse
from cascade.models import Project, ParamTypes
from neo4j.models import Neo4jGraph
from settings import logger

def add_arguments(parser):
    parser.add_argument("-p", "--project", type=str, dest="project",
                        help="project name")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Merge MEMM evidences based on the training set of the project given.')
    add_arguments(parser)
    args = parser.parse_args()

    project = Project(args.project)
    graph = Neo4jGraph('User')

    train_set, test_set = project.load_train_test()
    tsets = {}

    for cid in train_set:
        logger.info('loading MEMM evidences for cascade %s ...', cid)
        tsets_i = project.load_param('memm/evidence-{}'.format(cid), ParamTypes.JSON)

        logger.info('merging ...')
        if not tsets:
            tsets = tsets_i
        else:
            for uid in tsets_i:
                if uid not in tsets:
                    tsets[uid] = tsets_i[uid]
                else:
                    tsets[uid][1].extend(tsets_i[uid][1])

    logger.info('saving merged evidences ...')
    project.save_param(tsets, 'memm/evidence', ParamTypes.JSON)
