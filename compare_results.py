import argparse

from diffusion.enum import Method, Criterion
from cascade.models import Project
from cascade.testers import MultiProcTester
from config.predict_config import get_saved_params
from settings import logger
from utils.time_utils import time_measure


def run_method(method, project, init_depth, max_depth, eco, criterion):
    tester = MultiProcTester(project, method, criterion, eco=eco)
    params = get_saved_params(project.name, method)
    logger.debug('params = %s', params)
    _, _, res_trees = tester.run(init_depth, max_depth, **params)
    return res_trees


def compare(method1: Method, method2: Method, project_name: str, init_depth: int, max_depth: int, eco: bool,
            criterion: Criterion):
    project = Project(project_name)
    trees1 = run_method(method1, project, init_depth, max_depth, eco, criterion)
    trees2 = run_method(method2, project, init_depth, max_depth, eco, criterion)
    trees = project.load_trees()
    train_set, test_set = project.load_sets()
    orig_trees = [trees[cascade_id] for cascade_id in test_set]
    init_nodes = [set(tree.nodes(max_depth=init_depth)) for tree in orig_trees]
    output1 = [set(tree.nodes(max_depth=max_depth)) - init for tree, init in zip(trees1, init_nodes)]
    output2 = [set(tree.nodes(max_depth=max_depth)) - init for tree, init in zip(trees2, init_nodes)]
    true_output = [set(tree.nodes(max_depth=max_depth)) - init for tree, init in zip(orig_trees, init_nodes)]

    for i in range(len(test_set)):
        print(f'\nresults of cascade {test_set[i]}:')
        print_diff(method1, method2, output1[i], output2[i], true_output[i])
        print_diff(method2, method1, output2[i], output1[i], true_output[i])


def print_diff(method1, method2, output1, output2, true_output):
    """
    Print the nodes which are present in the results of method1 and absent in method2.
    """
    out_dict1 = {node.user_id: node for node in output1}
    out_dict2 = {node.user_id: node for node in output2}
    true_node_ids = set(node.user_id for node in true_output)
    print(f'\nnodes in results of {method1.value} but not in results of {method2.value}:')
    print(f'{"parent":<30}{"node":<30}{"probability":<15}{"true?":<10}')
    for node_id in set(out_dict1.keys()) - set(out_dict2.keys()):
        print(
            f'{str(out_dict1[node_id].parent_id):<30}{str(node_id):<30}{out_dict1[node_id].probability:<15.3}{node_id in true_node_ids:<10}')


@time_measure()
def main():
    parser = argparse.ArgumentParser('Test information diffusion prediction')
    parser.add_argument("-p", "--project", help="project name")
    parser.add_argument("--method1", required=True, choices=[e.value for e in Method],
                        help="first method to compare")
    parser.add_argument("--method2", required=True, choices=[e.value for e in Method],
                        help="second method to compare")
    parser.add_argument("-i", "--init-depth", type=int, dest="init_depth", default=0,
                        help="the maximum depth of the initial nodes")
    parser.add_argument("-d", "--max-depth", type=int, dest="max_depth",
                        help="the maximum depth of cascade prediction")
    parser.add_argument("-e", "--eco", action='store_true', default=False,
                        help="If this option is given, the prediction is done in economical mode e.t. Memory consumption "
                             "is decreased and data is stored in DB and loaded everytime needed instead of storing in "
                             "RAM. Otherwise, no data is stored in DB.")
    parser.add_argument("-C", "--criterion", choices=[e.value for e in Criterion], default="nodes",
                        help="the criterion on which the evaluation is done")
    args = parser.parse_args()
    method1, method2 = Method(args.method1), Method(args.method2)

    compare(method1, method2, args.project, args.init_depth, args.max_depth, args.eco, Criterion(args.criterion))


if __name__ == '__main__':
    main()
