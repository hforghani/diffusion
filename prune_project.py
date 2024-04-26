import argparse
from typing import Dict, List

from bson import ObjectId

import networkx

from cascade.models import Project, ActSequence, ParamTypes
from utils.time_utils import time_measure


def get_and_save_degrees(project_name, trees, graph) -> Dict[ObjectId, dict]:
    tree_graphs = {}
    print("extracting trees ...")
    for cid, tree in trees.items():
        tree_graph = networkx.DiGraph(tree.edges())
        tree_graph.add_nodes_from(tree.node_ids())
        tree_graphs[cid] = tree_graph

    print("extracting degrees ...")
    degrees = {}
    for node in graph:
        sum_in = sum(tree.in_degree(node) for tree in tree_graphs.values() if node in tree)
        sum_out = sum(tree.out_degree(node) for tree in tree_graphs.values() if node in tree)
        degrees[node] = {"in": sum_in, "out": sum_out, "total": sum_in + sum_out}
    sorted_nodes = sorted(degrees, key=lambda node: (degrees[node]["total"], degrees[node]["in"]))

    file_name = f"data/degrees-{project_name}.out"
    print(f"writing degrees into {file_name} ...")
    with open(file_name, "w") as f:
        f.write(f"node\t\t\t\t\t\tin\tout\ttotal\n")
        for node in sorted_nodes:
            degree = degrees[node]
            f.write(f"{node}\t{degree['in']}\t{degree['out']}\t{degree['total']}\n")

    return degrees


def save_pruned_trees(removable_nodes, trees, out_project):
    print("pruning trees ...")
    for tree in trees.values():
        for node_id in removable_nodes:
            node = tree.get_node(node_id)
            if node is not None:
                if node in tree.roots:
                    tree.roots.remove(node)
                else:
                    tree.get_node(node.parent_id).children.remove(node)
    print("removing 0-depth cascades ...")
    count = 0
    for cid in trees:
        if all(len(root.children) == 0 for root in trees[cid].roots):
            count += 1
            del trees[cid]
    print(f"{count} 0-depth cascades removed")
    print("saving trees ...")
    out_project.save_trees(trees)


def save_pruned_sequences(removable_nodes: List[ObjectId],
                          sequences: Dict[ObjectId, ActSequence],
                          out_project: Project):
    for seq in sequences.values():
        seq.users = [user for user in seq.users if user not in removable_nodes]
        seq.times = [seq.user_times[user] for user in seq.users]
        seq.user_times = {seq.users[i]: seq.times[i] for i in range(len(seq.users))}
        seq.max_t = max(seq.times)
    out_project.save_act_sequences(sequences)


def save_pruned_graphs(removable_nodes: List[ObjectId], project, out_project):
    graph_info = project.load_param("graph_info", ParamTypes.JSON)
    removable_nodes = list(map(str, removable_nodes))
    for fname, cascades in graph_info.items():
        graph = project.load_param(fname, ParamTypes.GRAPH)
        graph.remove_nodes_from(removable_nodes)
        out_project.save_param(graph, fname, ParamTypes.GRAPH)
    out_project.save_param(graph_info, "graph_info", ParamTypes.JSON)


@time_measure()
def main(project_name, out_project_name, min_degree):
    print("loading data ...")
    project = Project(project_name)
    trees = project.load_trees()
    train_set, test_set = project.load_sets()
    graph, sequences = project.load_or_extract_graph_seq()

    degrees = get_and_save_degrees(project_name, trees, graph)
    removable_nodes = [ObjectId(node) for node in degrees if degrees[node]["total"] < min_degree]
    print(f"Number of nodes to be removed: {len(removable_nodes)}")

    out_project = Project(out_project_name, db=project.db)

    save_pruned_trees(removable_nodes, trees, out_project)

    train_set = [cid for cid in train_set if cid in trees]
    test_set = [cid for cid in test_set if cid in trees]
    out_project.save_sets(train_set, test_set)

    print("pruning and saving the new graphs ...")
    save_pruned_graphs(removable_nodes, project, out_project)

    sequences = {cid: seq for cid, seq in sequences.items() if cid in trees}
    print("pruning and saving the new sequences ...")
    save_pruned_sequences(removable_nodes, sequences, out_project)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Remove the nodes with degree smaller than `min_degree` and save the new project.")
    parser.add_argument("-p", "--project", required=True, help="project name")
    parser.add_argument("-o", "--out_project", required=True, help="output project name")
    parser.add_argument("--min-degree", default=1, type=int, help="minimum degree")
    args = parser.parse_args()
    assert args.project != args.out_project
    main(args.project, args.out_project, args.min_degree)
