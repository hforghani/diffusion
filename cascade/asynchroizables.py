import pprint
import traceback
from typing import Tuple

from networkx import DiGraph

import settings
from cascade.avg import LTAvg
from cascade.enum import Method, Criterion
from cascade.models import Project
from cascade.aslt import AsLT

from cascade.validation import Validation
from log_levels import DEBUG_LEVELV_NUM
from memm.models import BinMEMMModel, TDMEMMModel, ReducedTDMEMMModel, ReducedBinMEMMModel, \
    ParentSensTDMEMMModel, LongParentSensTDMEMMModel
from mln.file_generators import FileCreator
from mln.models import MLN
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
    meas = Validation(res_output, true_output, ref_set)
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
    meas = Validation(res_output, true_output, ref_set)
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


def train_cascades(method, project, multi_processed=False, eco=False):
    model_classes = {
        Method.ASLT: AsLT,
        Method.AVG: LTAvg,
        Method.BIN_MEMM: BinMEMMModel,
        Method.TD_MEMM: TDMEMMModel,
        Method.REDUCED_TD_MEMM: ReducedTDMEMMModel,
        Method.REDUCED_BIN_MEMM: ReducedBinMEMMModel,
        Method.PARENT_SENS_TD_MEMM: ParentSensTDMEMMModel,
        Method.LONG_PARENT_SENS_TD_MEMM: LongParentSensTDMEMMModel,
    }
    # Create and train the model if needed.
    if method == Method.MLN_PRAC:
        model = MLN(project, method='edge', format=FileCreator.FORMAT_PRACMLN)
    elif method == Method.MLN_ALCH:
        model = MLN(project, method='edge', format=FileCreator.FORMAT_ALCHEMY2)
    elif method in model_classes:
        model_clazz = model_classes[method]
        model = model_clazz(project).fit(multi_processed=multi_processed, eco=eco)
    else:
        raise Exception('invalid method "%s"' % method.value)
    return model


def test_cascades_multiproc(cascade_ids: list, method: Method, project: Project, threshold: list, initial_depth: int,
                            max_depth: int, criterion: Criterion, trees: dict, graph: DiGraph, eco: bool, model=None) \
        -> Tuple[dict, dict, dict, dict, dict, dict]:
    try:
        if model is None and eco:
            # Train (or fetch the trained models from db) in each process due to the size limit for pickling in Python
            # multi-processing.
            model = train_cascades(method, project)
        return test_cascades(cascade_ids, method, model, threshold, initial_depth, max_depth, criterion, trees, graph)
    except:
        logger.error(traceback.format_exc())
        raise


def test_cascades(cascade_ids: list, method: Method, model, thresholds: list, initial_depth: int, max_depth: int,
                  criterion: Criterion, trees: dict, graph: DiGraph) \
        -> Tuple[dict, dict, dict, dict, dict, dict]:
    try:
        prp1_list = {thr: [] for thr in thresholds}
        prp2_list = {thr: [] for thr in thresholds}
        precisions = {thr: [] for thr in thresholds}
        recalls = {thr: [] for thr in thresholds}
        fprs = {thr: [] for thr in thresholds}
        f1s = {thr: [] for thr in thresholds}
        max_step = max_depth - initial_depth if max_depth is not None else None
        count = 1

        for cid in cascade_ids:
            tree = trees[cid]

            if initial_depth >= tree.depth:
                count += 1
                logger.info('cascade <%s> ignored since the initial depth is more than or equal to the tree depth', cid)
                continue

            logger.info('running prediction with method <%s> on cascade <%s>', method.value, cid)

            # Copy roots in a new tree.
            initial_tree = tree.copy(initial_depth)

            # Predict remaining nodes.
            with Timer('prediction', level='debug'):
                # TODO: apply max_depth for all methods.
                if method in [Method.MLN_PRAC, Method.MLN_ALCH]:
                    res_trees = model.predict(cid, initial_tree, threshold=thresholds)
                else:
                    res_trees = model.predict(initial_tree, graph, thresholds=thresholds, max_step=max_step)

            # Evaluate the results.
            with Timer('evaluating results', level='debug'):
                logs = [f'{"threshold":>10}{"output":>10}{"true":>10}{"precision":>10}{"recall":>10}{"f1":>10}']
                for thr in thresholds:
                    res_tree = res_trees[thr]

                    meas, res_output, true_output = evaluate(initial_tree, res_tree, tree, max_depth, criterion, graph)

                    if method in [Method.MLN_PRAC, Method.MLN_ALCH]:
                        prp = meas.prp(model.probabilities)
                        prp1 = prp[0] if prp else 0
                        prp2 = prp[1] if len(prp) > 1 else 0
                        prp1_list[thr].append(prp1)
                        prp2_list[thr].append(prp2)

                    prec = meas.precision()
                    rec = meas.recall()
                    fpr = meas.fpr()
                    f1 = meas.f1()
                    precisions[thr].append(prec)
                    recalls[thr].append(rec)
                    fprs[thr].append(fpr)
                    f1s[thr].append(f1)

                    logs.append(
                        f'{thr:10.3f}{len(res_output):10}{len(true_output):10}{prec:10.3f}{rec:10.3f}{f1:10.3f}')
                    # if method in ['aslt', 'avg']:
                    #     log += ', prp = (%.3f, %.3f, ...)' % (prp1, prp2)
                    # log_trees(tree, res_trees, max_depth)

                logger.debug(f'results of cascade {cid} ({count}/{len(cascade_ids)}) :\n' + '\n'.join(logs))

            count += 1

        return precisions, recalls, f1s, fprs, prp1_list, prp2_list

    except:
        logger.error(traceback.format_exc())
        raise
