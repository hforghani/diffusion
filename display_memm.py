import argparse
from pprint import PrettyPrinter
from bson import ObjectId
import numpy as np

from db.managers import EvidenceManager, MEMMManager
from cascade.models import Project
from cascade.enum import Method
from memm.memm import array_to_str, ParentTDMEMM


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)

def print_info(user_id, evidences, memm, graph):
    dim = evidences['dimension']
    pp = PrettyPrinter()
    new_sequences, orig_indexes = memm.decrease_dim(evidences['sequences'], dim)
    str_sequences = [[(array_to_str(obs), state) for obs, state in seq] for seq in new_sequences]
    parents = list(graph.predecessors(user_id))
    parent_user_ids = [parents[index] for index in orig_indexes]

    print('Lambda:')
    pp.pprint(list(memm.Lambda))
    print('\nActivation probability of observations:')
    all_obs, _ = memm.get_all_obs_mat(new_sequences)
    print_probs(all_obs, memm, parents)
    print('\nSelected indexes:')
    pp.pprint(orig_indexes)
    print('\nUser ids of selected indexes:')
    pp.pprint(parent_user_ids)
    print('\nEvidences with decreased dimensions:')
    pp.pprint(str_sequences)
    # print('\nEvidences:')
    # pp.pprint([[(obs_to_str(obs, dim), state) for obs, state in seq] for seq in evidences['sequences']])


def print_probs(all_obs, memm, parents):
    new_dim = len(memm.orig_indexes)

    if isinstance(memm, ParentTDMEMM):
        all_states = list(range(len(parents) + 1))
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

    print(f"{'decreased vector':<{(new_dim + 1) * 5}}" + ''.join([f"{s:<10}" for s in states]))
    for i in range(all_obs.shape[0]):
        obs = all_obs[i, :]
        print(f"{array_to_str(obs):<{(new_dim + 1) * 5}}" + ''.join([f"{p:<10.5f}" for p in probs[i, :]]))


def handle():
    parser = argparse.ArgumentParser('Display evidences of a user id')
    parser.add_argument('-p', '--project', type=str, help='project name', required=True)
    parser.add_argument('-u', '--userid', dest='user_id', type=str, help='user id', required=True)
    parser.add_argument('-m', '--method', type=str, help='MEMM method', required=True)
    args = parser.parse_args()

    methods = {e.value: e for e in Method}
    project = Project(args.project)
    memm = MEMMManager(project, methods[args.method]).fetch_one(args.user_id)
    m = EvidenceManager(project, methods[args.method])
    evidences = m.get_one(ObjectId(args.user_id))
    graph = project.load_or_extract_graph()
    print_info(ObjectId(args.user_id), evidences, memm, graph)


if __name__ == '__main__':
    handle()
