import argparse
import pprint
from functools import reduce

from bson import ObjectId

from db.managers import SeqLabelDBManager, ParentSensEvidManager
from cascade.models import Project
from diffusion.enum import Method
from seq_labeling.utils import obs_to_str, arr_to_str


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
    states = [[state for obs, state in seq] for seq in sequences]
    states = reduce(lambda l1, l2: l1 + l2, states)
    features = memm.sequences_to_feat_states(sequences)
    dim = features.shape[1]
    all_states = list(range(len(dim_users) + 1))
    # all_states = [False, True]
    probs = memm.get_multi_obs_probs(sequences)

    states_to_print = all_states
    # states_to_print = [True]
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
        dim_users = sorted(graph.nodes())
    else:
        dim_users = list(graph.predecessors(key))
    return dim_users


def handle():
    parser = argparse.ArgumentParser('Display evidences of a user id')
    parser.add_argument('-p', '--project', type=str, help='project name', required=True)
    parser.add_argument('-u', '--userid', dest='user_id', type=str, help='user id (for node MEMMs)', required=True)
    parser.add_argument('-m', '--method', type=str, help='MEMM method', required=True)
    args = parser.parse_args()

    methods = {e.value: e for e in Method}
    method = methods[args.method]
    project = Project(args.project)
    manager = SeqLabelDBManager(project, method)
    evid_manager = ParentSensEvidManager(project)
    # evid_manager = EvidenceManager(project)
    key = ObjectId(args.user_id)

    memm = manager.fetch_one(key)
    if memm is None:
        parser.error('MEMM does not exist')
    evidences = evid_manager.get_one(key)
    graph = project.load_or_extract_graph()
    print_info(key, evidences, memm, graph, method)


if __name__ == '__main__':
    handle()
