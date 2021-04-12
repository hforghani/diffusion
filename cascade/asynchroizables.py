import traceback

from cascade.avg import LTAvg
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
    logger.debug('len(all_nodes) = %d', len(all_nodes))

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


def train_memes(method, project):
    # Create and train the model if needed.
    if method == 'mlnprac':
        model = MLN(project, method='edge', format=FileCreator.FORMAT_PRACMLN)
    elif method == 'mlnalch':
        model = MLN(project, method='edge', format=FileCreator.FORMAT_ALCHEMY2)
    elif method == 'memm':
        train_set, _, _ = project.load_sets()
        model = MEMMModel(project).fit(train_set)
    elif method == 'aslt':
        model = Saito(project)
    elif method == 'avg':
        model = LTAvg(project)
    else:
        raise Exception('invalid method "%s"' % method)
    return model


def test_memes_multiproc(meme_ids, method, project, threshold, initial_depth, max_depth, trees, all_node_ids, user_ids,
                         users_map):
    try:
        model = train_memes(method, project)
        return test_memes(meme_ids, method, model, threshold, initial_depth, max_depth, trees, all_node_ids, user_ids,
                          users_map)
    except:
        logger.error(traceback.format_exc())
        raise


def test_memes(meme_ids, method, model, threshold, initial_depth, max_depth, trees, all_node_ids, user_ids, users_map):
    try:
        prp1_list = []
        prp2_list = []
        precisions = []
        recalls = []
        fprs = []
        f1s = []
        max_step = max_depth - initial_depth if max_depth is not None else None
        count = 1

        for meme_id in meme_ids:
            with Timer('getting tree'):
                tree = trees[meme_id]

            # Copy roots in a new tree.
            with Timer('copying tree'):
                # logger.debugv('\n' + tree.render(digest=True))
                initial_tree = tree.copy(initial_depth)

            # Predict remaining nodes.
            with Timer('prediction'):
                logger.info('running prediction with method <%s> on meme <%s>', method, meme_id)
                # TODO: apply max_depth for all methods.
                if method in ['mlnprac', 'mlnalch']:
                    res_tree = model.predict(meme_id, initial_tree, threshold=threshold)
                elif method in ['aslt', 'avg']:
                    res_tree = model.predict(initial_tree, threshold=threshold, max_step=max_step, user_ids=user_ids,
                                             users_map=users_map)
                elif method == 'memm':
                    res_tree = model.predict(initial_tree, threshold=threshold, max_step=max_step, multiprocessed=False)

            # Evaluate the results.
            with Timer('evaluating results'):
                meas, res_output, true_output = evaluate(initial_tree, res_tree, tree, all_node_ids, max_depth)

            if method in ['aslt', 'avg']:
                prp = meas.prp(model.probabilities)
                prp1 = prp[0] if prp else 0
                prp2 = prp[1] if len(prp) > 1 else 0
                prp1_list.append(prp1)
                prp2_list.append(prp2)

            with Timer('reporting results'):
                prec = meas.precision()
                rec = meas.recall()
                fpr = meas.fpr()
                f1 = meas.f1()
                precisions.append(prec)
                recalls.append(rec)
                fprs.append(fpr)
                f1s.append(f1)

                log = f'meme {meme_id} ({count}/{len(meme_ids)}): {len(res_output)} outputs, ' \
                          f'{len(true_output)} true, precision = {prec:.3f}, recall = {rec:.3f}, f1 = {f1:.3f}'
                if method in ['aslt', 'avg']:
                    log += ', prp = (%.3f, %.3f, ...)' % (prp1, prp2)
                logger.info(log)
                # Notice: This line takes too much execution time:
                # log_trees(tree, res_tree, max_depth)
                count += 1

        return precisions, recalls, f1s, fprs, prp1_list, prp2_list

    except:
        logger.error(traceback.format_exc())
        raise
