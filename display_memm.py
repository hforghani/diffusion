import argparse
import pprint
from functools import reduce

from bson import ObjectId
import numpy as np

from db.managers import EvidenceManager, MEMMManager, EdgeMEMMManager, EdgeEvidenceManager, ParentSensEvidManager
from cascade.models import Project
from diffusion.enum import Method
from memm.memm import obs_to_str, ParentTDMEMM, arr_to_str


def print_info(key, evidences, memm, graph, method):
    str_sequences = [[(obs_to_str(obs), state) for obs, state in seq] for seq in evidences['sequences']]
    dim_users = get_dim_users(key, graph, method)

    print_lambda(memm)
    print('\nOriginal indexes:')
    pprint.pprint(memm.orig_indexes)
    print()
    print_probs(evidences['sequences'], memm, dim_users)
    print('\nEvidences:')
    pprint.pprint(str_sequences)
    # print('\nEvidences:')
    # pprint.pprint([[(array_to_str(obs), state) for obs, state in seq] for seq in evidences['sequences']])


def print_lambda(memm):
    print(f'{"index":<10}{"lambda":<10}')
    for i in range(memm.Lambda.size):
        print(f'{i:<10}{memm.Lambda[i]:<10}')


def print_probs(sequences, memm, dim_users):
    observations = [[obs for obs, state in seq] for seq in sequences]
    observations = reduce(lambda l1, l2: l1 + l2, observations)
    features, states = memm.sequences_to_feat_states(sequences)
    dim = features.shape[1]
    if isinstance(memm, ParentTDMEMM):
        all_states = list(range(len(dim_users) + 1))
    else:
        all_states = [False, True]
    probs = memm.get_multi_obs_probs(observations, all_states)

    if isinstance(memm, ParentTDMEMM):
        states_to_print = all_states
    else:
        states_to_print = [True]
    state_indexes = [all_states.index(s) for s in states_to_print]

    col0_width = max((dim + 1) * 5, 30)
    print(f"{'feature':<{col0_width}}" + f"{'state':<10}" + ''.join([f"{s:<10}" for s in states_to_print]))
    for i in range(features.shape[0]):
        print(f"{arr_to_str(features[i, :]):<{col0_width}}" + f"{states[i]:<10}" + ''.join(
            [f"{p:<10.5f}" for p in probs[i, state_indexes]]))


def get_dim_users(key, graph, method):
    if method in [Method.TD_EDGE_MEMM]:
        src_children = graph.successors(key[0])
        dest_parents = graph.predecessors(key[1])
        dim_users = sorted(set(src_children) | set(dest_parents) - {key[1]})
    elif method == Method.FULL_TD_MEMM:
        dim_users = list(graph.nodes())
    else:
        dim_users = list(graph.predecessors(key))
    return dim_users


def handle():
    parser = argparse.ArgumentParser('Display evidences of a user id')
    parser.add_argument('-p', '--project', type=str, help='project name', required=True)
    parser.add_argument('-u', '--userid', dest='user_id', type=str, help='user id (for node MEMMs)')
    parser.add_argument('-s', '--src', dest='src', type=str, help='source user id (for edge MEMMs)')
    parser.add_argument('-d', '--dst', dest='dst', type=str, help='destination user id (for edge MEMMs)')
    parser.add_argument('-m', '--method', type=str, help='MEMM method', required=True)
    args = parser.parse_args()

    methods = {e.value: e for e in Method}
    method = methods[args.method]
    project = Project(args.project)
    if args.user_id:
        manager = MEMMManager(project, method)
        if method in [Method.PARENT_SENS_TD_MEMM, Method.LONG_PARENT_SENS_TD_MEMM]:
            evid_manager = ParentSensEvidManager(project)
        else:
            evid_manager = EvidenceManager(project)
        EvidenceManager(project)
        key = ObjectId(args.user_id)
    else:
        manager = EdgeMEMMManager(project, method)
        evid_manager = EdgeEvidenceManager(project)
        key = (ObjectId(args.src), ObjectId(args.dst))

    memm = manager.fetch_one(key)
    if memm is None:
        parser.error('MEMM does not exist')
    evidences = evid_manager.get_one(key)
    graph = project.load_or_extract_graph()
    print_info(key, evidences, memm, graph, method)


if __name__ == '__main__':
    handle()
