import traceback
import typing
from typing import Tuple, Any

from networkx import DiGraph

import settings
from cascade.metric import Metric
from diffusion.enum import Method, Criterion
from log_levels import DEBUG_LEVELV_NUM
from memm.models import TDEdgeMEMMModel
from settings import logger
from utils.time_utils import Timer


def evaluate(initial_tree, res_tree, tree, max_depth, criterion, graph):
    if criterion == Criterion.NODES:
        all_node_ids = list(graph.nodes())
        meas, res_output, true_output = evaluate_nodes(initial_tree, res_tree, tree, all_node_ids,
                                                       max_depth)
    else:
        all_edges = set(graph.edges())
        meas, res_output, true_output = evaluate_edges(initial_tree, res_tree, tree, all_edges, max_depth)
    return meas, res_output, true_output


def evaluate_nodes(initial_tree, res_tree, tree, all_nodes, max_depth=None):
    # Get predicted and true nodes.
    res_nodes = set(res_tree.node_ids())
    true_nodes = set(tree.node_ids(max_depth=max_depth))
    initial_nodes = set(initial_tree.node_ids())
    res_output = res_nodes - initial_nodes
    true_output = true_nodes - initial_nodes
    ref_set = set(all_nodes) - initial_nodes

    # Evaluate the result.
    meas = Metric(res_output, true_output, ref_set)
    return meas, res_output, true_output


def evaluate_edges(initial_tree, res_tree, tree, all_edges, max_depth=None):
    # Get predicted and true nodes.
    res_edges = set(res_tree.edges(max_depth=max_depth))
    true_edges = set(tree.edges(max_depth=max_depth))
    initial_edges = set(initial_tree.edges())
    res_output = res_edges - initial_edges
    true_output = true_edges - initial_edges
    ref_set = set(all_edges) - initial_edges

    # Evaluate the result.
    meas = Metric(res_output, true_output, ref_set)
    return meas, res_output, true_output


def log_trees(tree, res_trees, max_depth=None, level=DEBUG_LEVELV_NUM):
    # TODO: res_trees is a dictionary of thresholds to trees.
    if settings.LOG_LEVEL <= level:
        if max_depth is not None:
            tree = tree.copy(max_depth)
        tree_render = tree.render(digest=True).split('\n')
        res_tree_render = res_trees.render(digest=True).split('\n')
        max_len = max([len(line) for line in tree_render])
        max_lines = max(len(tree_render), len(res_tree_render))
        formatted_line = '{:' + str(max_len + 5) + '}{}'
        logger.log(level, formatted_line.format('true tree:', 'output tree:'))
        for i in range(max_lines):
            logger.log(level, formatted_line.format(tree_render[i] if i < len(tree_render) else '',
                                                    res_tree_render[i] if i < len(res_tree_render) else ''))


def test_cascades(cascade_ids: list, method: Method, model, thresholds: Any, initial_depth: int, max_depth: int,
                  criterion: Criterion, trees: dict, graph: DiGraph) -> typing.Union[dict, list]:
    try:
        results = {thr: [] for thr in thresholds} if isinstance(thresholds, list) else []
        max_step = max_depth - initial_depth if max_depth is not None else None
        count = 1

        if method == Method.TD_EDGE_MEMM:
            dim_user_indexes_map = TDEdgeMEMMModel.extract_dim_user_indexes_map(graph)
        else:
            dim_user_indexes_map = None

        for cid in cascade_ids:
            tree = trees[cid]

            if initial_depth >= tree.depth:
                count += 1
                logger.info('cascade <%s> ignored since the initial depth is more than or equal to the tree depth', cid)
                if isinstance(thresholds, list):
                    for thr in thresholds:
                        results[thr].append(None)
                else:
                    results.append(None)
            else:
                logger.info('running prediction with method <%s> on cascade <%s>', method.value, cid)

                # Copy roots in a new tree.
                initial_tree = tree.copy(initial_depth)

                # Predict remaining nodes.
                with Timer('prediction', level='debug'):
                    # TODO: apply max_depth for all methods.
                    if method in [Method.MLN_PRAC, Method.MLN_ALCH]:
                        res_trees = model.predict(cid, initial_tree, threshold=thresholds)
                    elif method == Method.TD_EDGE_MEMM:
                        res_trees = model.predict(initial_tree, graph, thresholds=thresholds, max_step=max_step,
                                                  dim_user_indexes_map=dim_user_indexes_map)
                    else:
                        res_trees = model.predict(initial_tree, graph, thresholds=thresholds, max_step=max_step)

                # Evaluate the results.
                with Timer('evaluating results', level='debug'):
                    if isinstance(thresholds, list):
                        logs = [f'{"threshold":>10}{"output":>10}{"true":>10}{"precision":>10}{"recall":>10}{"f1":>10}']
                        for thr in thresholds:
                            res_tree = res_trees[thr]
                            meas, res_output, true_output = evaluate(initial_tree, res_tree, tree, max_depth, criterion,
                                                                     graph)
                            results.append(meas)
                            logs.append(
                                f'{thr:10.3f}{len(res_output):10}{len(true_output):10}{meas.precision():10.3f}'
                                f'{meas.recall():10.3f}{meas.f1():10.3f}')
                    else:
                        # res_trees is an instance of CascadeTree
                        logs = [f'{"output":>10}{"true":>10}{"precision":>10}{"recall":>10}{"f1":>10}']
                        meas, res_output, true_output = evaluate(initial_tree, res_trees, tree, max_depth, criterion,
                                                                 graph)
                        results.append(meas)
                        logs.append(
                            f'{len(res_output):10}{len(true_output):10}{meas.precision():10.3f}{meas.recall():10.3f}'
                            f'{meas.f1():10.3f}')

                    # log_trees(tree, res_trees, max_depth)
                    logger.debug(f'results of cascade {cid} ({count}/{len(cascade_ids)}) :\n' + '\n'.join(logs))

            count += 1

        return results

    except:
        logger.error(traceback.format_exc())
        raise
