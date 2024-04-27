from __future__ import annotations
import traceback
from typing import Tuple, List, Dict

from networkx import DiGraph
from pympler.asizeof import asizeof
from mlstatpy.graph import GraphDistance

import settings
from cascade.metric import Metric
from cascade.models import CascadeTree
from config.predict_config import PredictConfig
from diffusion.enum import Method, Criterion
from log_levels import DEBUG_LEVELV_NUM
from settings import logger
from utils.time_utils import Timer


def graph_distance(tree1: CascadeTree, tree2: CascadeTree) -> float:
    """
    Calculate the graph distance between two trees.
    Assumption: The roots of both trees are same.
    """
    # Add a new root to the both trees to ensure the trees have edges.
    edges1 = ([("root", str(root.user_id)) for root in tree1.roots] +
              [(str(n1), str(n2)) for n1, n2 in tree1.edges()])
    edges2 = ([("root", str(root.user_id)) for root in tree2.roots] +
              [(str(n1), str(n2)) for n1, n2 in tree2.edges()])
    graph1 = GraphDistance(edges1)
    graph2 = GraphDistance(edges2)
    distance, graph = graph1.distance_matching_graphs_paths(graph2, use_min=False)
    return distance


def evaluate_nodes(initial_tree, res_tree, tree, graph: DiGraph, max_depth=None):
    # Get predicted and true nodes.
    res_nodes = set(res_tree.node_ids())
    true_nodes = set(tree.node_ids(max_depth=max_depth))
    initial_nodes = set(initial_tree.node_ids())
    res_output = res_nodes - initial_nodes
    true_output = true_nodes - initial_nodes
    # succ_lists = [networkx.dfs_successors(graph, node) for node in initial_tree.node_ids() if node in graph]
    # all_nodes = set().union(*succ_lists) | true_nodes
    # ref_set = set(all_nodes) - initial_nodes
    ref_set = set(graph.nodes()) | true_nodes - initial_nodes

    # Evaluate the result.
    meas = Metric(res_output, true_output, ref_set)
    return meas, res_output, true_output


def evaluate(initial_tree, res_tree, tree, max_depth, criterion, graph, on_test=False) -> Tuple[Metric, set, set]:
    """
    @param initial_tree : CascadeTree including initial nodes
    @param res_tree : CascadeTree including result of prediction
    @param tree : CascadeTree including true result
    @param max_depth : maximum depth of prediction
    @param criterion : evaluation criterion: nodes or edges
    @param graph : directed graph of training nodes
    @param on_test : whether it is on test stage
    """
    if criterion == Criterion.NODES:
        meas, res_output, true_output = evaluate_nodes(initial_tree, res_tree, tree, graph, max_depth)
    else:
        meas, res_output, true_output = evaluate_edges(initial_tree, res_tree, tree, graph, max_depth)
    if on_test and "graph_dist" in PredictConfig().additional_metrics:
        meas["graph_dist"] = graph_distance(res_tree, tree)
    return meas, res_output, true_output


def evaluate_edges(initial_tree, res_tree, tree, graph: DiGraph, max_depth=None):
    # Get predicted and true nodes.
    res_edges = set(res_tree.edges(max_depth=max_depth))
    true_edges = set(tree.edges(max_depth=max_depth))
    initial_edges = set(initial_tree.edges())
    res_output = res_edges - initial_edges
    true_output = true_edges - initial_edges
    # edge_lists = [networkx.dfs_edges(graph, node) for node in initial_tree.node_ids() if node in graph]
    # all_edges = set().union(*edge_lists) | true_edges
    # ref_set = set(all_edges) - initial_edges
    ref_set = set(graph.edges()) | true_edges - initial_edges

    # Evaluate the result.
    meas = Metric(res_output, true_output, ref_set)
    return meas, res_output, true_output


def log_trees(tree, res_trees, max_depth=None, level=DEBUG_LEVELV_NUM):
    # TODO: res_trees is a dictionary of thresholds to trees.
    if settings.LOG_LEVEL <= level:
        if max_depth is not None:
            tree = tree.copy(max_depth)
        tree_render = tree.render().split('\n')
        res_tree_render = res_trees.render().split('\n')
        max_len = max([len(line) for line in tree_render])
        max_lines = max(len(tree_render), len(res_tree_render))
        formatted_line = '{:' + str(max_len + 5) + '}{}'
        logger.log(level, formatted_line.format('true tree:', 'output tree:'))
        for i in range(max_lines):
            logger.log(level, formatted_line.format(tree_render[i] if i < len(tree_render) else '',
                                                    res_tree_render[i] if i < len(res_tree_render) else ''))


def test_cascades(cascade_ids: list, method: Method, model, initial_depth: int, max_depth: int, criterion: Criterion,
                  trees: dict, graph: DiGraph, on_test: bool, threshold: list | float, **params) -> Tuple[
    List[Metric] | Dict[float, List[Metric]],
    List[CascadeTree]
]:
    """
    @param on_test : whether it is on test stage
    """
    try:
        logger.debug('type(threshold) = %s', type(threshold))
        results = {thr: [] for thr in threshold} if isinstance(threshold, list) else []
        res_trees = [] if not isinstance(threshold, list) else None  # Only used when on test stage
        max_step = max_depth - initial_depth if max_depth is not None else None
        count = 1

        for cid in cascade_ids:
            tree = trees[cid]

            if initial_depth >= tree.depth:
                count += 1
                logger.info('cascade <%s> ignored since the initial depth is more than or equal to the tree depth', cid)
                if isinstance(threshold, list):
                    for thr in threshold:
                        results[thr].append(None)
                else:
                    results.append(None)
            else:
                logger.debug('running prediction with method <%s> on cascade <%s>', method.value, cid)

                # Copy roots in a new tree.
                initial_tree = tree.copy(initial_depth)

                # Predict remaining nodes.
                with Timer('prediction', level='debug'):
                    res_tree = model.predict_one_sample(initial_tree, threshold, graph, max_step)

                # Evaluate the results.
                with Timer('evaluating results', level='debug'):
                    if isinstance(threshold, list):
                        logs = [f'{"threshold":>10}{"output":>10}{"true":>10}{"precision":>10}{"recall":>10}{"f1":>10}']
                        for thr in threshold:
                            meas, res_output, true_output = evaluate(initial_tree, res_tree[thr], tree, max_depth,
                                                                     criterion, graph, on_test)
                            results[thr].append(meas)
                            logs.append(
                                f'{thr:10.3f}{len(res_output):10}{len(true_output):10}' + "".join(
                                    f'{meas[metric]:10.3f}' for metric in meas.metrics)
                            )
                    else:
                        # res_tree is an instance of CascadeTree
                        logs = [f'{"output":>10}{"true":>10}{"precision":>10}{"recall":>10}{"f1":>10}']
                        meas, res_output, true_output = evaluate(initial_tree, res_tree, tree, max_depth, criterion,
                                                                 graph, on_test)
                        results.append(meas)
                        res_trees.append(res_tree)
                        logs.append(
                            f'{len(res_output):10}{len(true_output):10}' + "".join(
                                f'{meas[metric]:10.3f}' for metric in meas.metrics)
                        )

                    # log_trees(tree, res_tree, max_depth)
                    logger.debug(f'results of cascade {cid} ({count}/{len(cascade_ids)}) :\n' + '\n'.join(logs))

            count += 1

        # logger.debug('sizes in test_cascades (MB):')
        # for key, value in locals().items():
        #     logger.debug(f'{key:<30}{asizeof(value) / (1024 ** 2)}')

        logger.info('done')
        logger.debug('type(results) = %s', type(results))
        return results, res_trees

    except:
        logger.error(traceback.format_exc())
        raise
