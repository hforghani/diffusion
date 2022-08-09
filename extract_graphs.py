import argparse
import itertools

from networkx import DiGraph

from cascade.models import Project, ParamTypes
from cascade.testers import DefaultTester
from db.managers import DBManager
from settings import logger


def save_graph(project, db, cascade_ids):
    graph_info_fname = 'graph_info'
    try:
        graph_info = project.load_param(graph_info_fname, ParamTypes.JSON)
    except FileNotFoundError:
        graph_info = {}
    if not graph_info:
        fname = 'graph1'
    else:
        fname = 'graph' + str(max(int(name[5:]) for name in graph_info.keys()) + 1)

    logger.info('reading postcascades ...')
    post_ids = {pc['post_id'] for pc in
                db.postcascades.find({'cascade_id': {'$in': cascade_ids}}, ['post_id'])}
    logger.info('%d posts collected', len(post_ids))
    logger.info('reading reshares ...')
    edges = {(resh['ref_user_id'], resh['user_id']) for resh in db.reshares.find() if resh['post_id'] in post_ids}
    graph = DiGraph()
    graph.add_edges_from(edges)

    logger.info('saving graph ...')
    project.save_param(graph, fname, ParamTypes.GRAPH)
    graph_info[fname] = [str(cid) for cid in cascade_ids]
    project.save_param(graph_info, graph_info_fname, ParamTypes.JSON)


def extract_graphs(db, project_name):
    project = Project(project_name)
    folds_num = 3
    folds = DefaultTester(project, None).get_cross_val_folds(folds_num)
    logger.info('extracting graphs of %d folds if not exist ...', folds_num)

    for i in range(folds_num):
        logger.info('extracting graph missing fold %d ...', i + 1)
        train_set = list(itertools.chain(*(folds[:i] + folds[i + 1:])))
        save_graph(project, db, train_set)


def main(db_name, project_name):
    manager = DBManager(db_name)
    db = manager.db
    extract_graphs(db, project_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract the graphs for the db in which each post belongs to one cascade.')
    parser.add_argument("-p", "--project", required=True, help="project name")
    parser.add_argument('-d', '--db', required=True, help="db name in which the documents must be inserted")
    args = parser.parse_args()
    main(args.db, args.project)
