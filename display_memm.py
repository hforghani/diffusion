import argparse
import pprint
from bson import ObjectId
import numpy as np

from db.managers import EvidenceManager, MEMMManager, EdgeMEMMManager, EdgeEvidenceManager
from cascade.models import Project
from diffusion.enum import Method
from memm.memm import array_to_str, ParentTDMEMM


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)

def print_info(key, evidences, memm, graph, method):
    dim = evidences['dimension']
    new_sequences, orig_indexes = memm.decrease_dim(evidences['sequences'], dim)
    str_sequences = [[(array_to_str(obs), state) for obs, state in seq] for seq in new_sequences]
    dim_users = get_dim_users(key, graph, method)
    dec_dim_users = [dim_users[index] for index in orig_indexes]

    print_lambda(memm, orig_indexes, dec_dim_users)
    print('\nActivation probability of observations:')
    all_obs, _ = memm.get_all_obs_mat(new_sequences)
    print_probs(all_obs, memm, dim_users)
    print('\nEvidences with decreased dimensions:')
    pprint.pprint(str_sequences)
    # print('\nEvidences:')
    # pprint.pprint([[(array_to_str(obs), state) for obs, state in seq] for seq in evidences['sequences']])


def print_lambda(memm, orig_indexes, dec_dim_users):
    print(f'{"index":<10}{"user id":<30}{"lambda":<10}')
    for i in range(len(orig_indexes)):
        print(f'{orig_indexes[i]:<10}{str(dec_dim_users[i]):<30}{memm.Lambda[i]:<10}')


def print_probs(all_obs, memm, dim_users):
    new_dim = len(memm.orig_indexes)

    if isinstance(memm, ParentTDMEMM):
        all_states = list(range(len(dim_users) + 1))
        states = [0] + [i + 1 for i in memm.orig_indexes]
        probs = np.zeros((all_obs.shape[0], len(states) + 1))
        for i in range(all_obs.shape[0]):
            obs = all_obs[i, :]
            probs[i, :] = memm.get_probs(obs, all_states)
    else:
        states = [True]
        probs = np.zeros((all_obs.shape[0], 1))
        for i in range(all_obs.shape[0]):
            obs = all_obs[i, :]
            probs[i] = memm.get_prob(obs, True, [False, True])

    col0_width = min((new_dim + 1) * 5, 30)
    print(f"{'decreased vector':<{col0_width}}" + ''.join([f"{s:<10}" for s in states]))
    for i in range(all_obs.shape[0]):
        obs = all_obs[i, :]
        print(f"{array_to_str(obs):<{col0_width}}" + ''.join([f"{p:<10.5f}" for p in probs[i, :]]))


def get_dim_users(key, graph, method):
    if method in [Method.TD_EDGE_MEMM]:
        src_children = graph.successors(key[0])
        dest_parents = graph.predecessors(key[1])
        dim_users = sorted(set(src_children) | set(dest_parents) - {key[1]})
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
        evid_manager = EvidenceManager(project, method)
        key = ObjectId(args.user_id)
    else:
        manager = EdgeMEMMManager(project, method)
        evid_manager = EdgeEvidenceManager(project, method)
        key = (ObjectId(args.src), ObjectId(args.dst))

    memm = manager.fetch_one(key)
    if memm is None:
        parser.error('MEMM does not exist')
    evidences = evid_manager.get_one(key)
    graph = project.load_or_extract_graph()
    print_info(key, evidences, memm, graph, method)


if __name__ == '__main__':
    handle()
