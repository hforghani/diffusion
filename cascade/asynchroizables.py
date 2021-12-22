import traceback
from typing import Tuple

from cascade.avg import LTAvg
from cascade.models import Project
from cascade.saito import Saito

from cascade.validation import Validation
from memm.models import MEMMModel
from mln.file_generators import FileCreator
from mln.models import MLN
from settings import logger
from utils.time_utils import Timer


def evaluate(initial_tree, res_tree, tree, all_nodes, max_depth=None):
    # Get predicted and true nodes.
    res_nodes = set(res_tree.node_ids())
    true_nodes = set(tree.node_ids(max_depth=max_depth))
    initial_nodes = set(initial_tree.node_ids())
    res_output = res_nodes - initial_nodes
    true_output = true_nodes - initial_nodes

    # Evaluate the result.
    meas = Validation(res_output, true_output, all_nodes)
    return meas, res_output, true_output


def log_trees(tree, res_tree, max_depth=None):
    if max_depth is not None:
        tree = tree.copy(max_depth)
    tree_render = tree.render(digest=True).split('\n')
    res_tree_render = res_tree.render(digest=True).split('\n')
    max_len = max([len(line) for line in tree_render])
    max_lines = max(len(tree_render), len(res_tree_render))
    formatted_line = '{:' + str(max_len + 5) + '}{}'
    logger.debugv(formatted_line.format('true tree:', 'output tree:'))
    for i in range(max_lines):
        logger.debugv(formatted_line.format(tree_render[i] if i < len(tree_render) else '',
                                            res_tree_render[i] if i < len(res_tree_render) else ''))


def train_cascades(method, project, multi_processed=False):
    # Create and train the model if needed.
    if method == 'mlnprac':
        model = MLN(project, method='edge', format=FileCreator.FORMAT_PRACMLN)
    elif method == 'mlnalch':
        model = MLN(project, method='edge', format=FileCreator.FORMAT_ALCHEMY2)
    elif method == 'memm':
        train_set, _, _ = project.load_sets()
        model = MEMMModel(project).fit(train_set, multi_processed)
    elif method == 'aslt':
        model = Saito(project)
    elif method == 'avg':
        model = LTAvg(project)
    else:
        raise Exception('invalid method "%s"' % method)
    return model


def test_cascades_multiproc(cascade_ids: list, method, project: Project, threshold: list, initial_depth: int,
                            max_depth: int, trees: dict, all_node_ids: list, user_ids: list, users_map: dict) \
        -> Tuple[dict, dict, dict, dict, dict, dict]:
    try:
        # Train (or fetch trained models from db) in each process due to size limit for pickling in Python
        # multi-processing.
        model = train_cascades(method, project)
        return test_cascades(cascade_ids, method, model, threshold, initial_depth, max_depth, trees, all_node_ids,
                             user_ids, users_map)
    except:
        logger.error(traceback.format_exc())
        raise


def test_cascades(cascade_ids: list, method: str, model, thresholds: list, initial_depth: int, max_depth: int,
                  trees: dict, all_node_ids: list, user_ids: list, users_map: dict) \
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

            logger.info('running prediction with method <%s> on cascade <%s>', method, cid)

            # Copy roots in a new tree.
            initial_tree = tree.copy(initial_depth)

            # Predict remaining nodes.
            with Timer('prediction', level='debug'):
                # TODO: apply max_depth for all methods.
                if method in ['mlnprac', 'mlnalch']:
                    res_trees = model.predict(cid, initial_tree, threshold=thresholds)
                elif method in ['aslt', 'avg']:
                    res_trees = model.predict(initial_tree, thresholds=thresholds, max_step=max_step,
                                              user_ids=user_ids, users_map=users_map)
                elif method == 'memm':
                    res_trees = model.predict(initial_tree, thresholds=thresholds, max_step=max_step,
                                              multiprocessed=False)

            # Evaluate the results.
            with Timer('evaluating results', level='debug'):
                logs = [f'{"threshold":>10}{"output":>10}{"true":>10}{"precision":>10}{"recall":>10}{"f1":>10}']
                for thr in thresholds:
                    res_tree = res_trees[thr]
                    meas, res_output, true_output = evaluate(initial_tree, res_tree, tree, all_node_ids, max_depth)

                    if method in ['aslt', 'avg']:
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
                        f'{thr:10.2f}{len(res_output):10}{len(true_output):10}{prec:10.3f}{rec:10.3f}{f1:10.3f}')
                    # if method in ['aslt', 'avg']:
                    #     log += ', prp = (%.3f, %.3f, ...)' % (prp1, prp2)
                    # log_trees(tree, res_trees, max_depth) # Notice: This line takes too much execution time:

                logger.info(f'results of cascade {cid} ({count}/{len(cascade_ids)}) :\n' + '\n'.join(logs))

            count += 1

        return precisions, recalls, f1s, fprs, prp1_list, prp2_list

    except:
        logger.error(traceback.format_exc())
        raise
