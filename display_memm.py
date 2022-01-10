import argparse
from functools import reduce
from pprint import PrettyPrinter
from bson import ObjectId
import numpy as np

from db.managers import EvidenceManager, DBManager, MEMMManager
from cascade.models import Project
from memm.enum import MEMMMethod
from memm.memm import MEMM, array_to_str, BinMEMM


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)

def print_info(user_id, evidences, memm, graph):
    dim = evidences['dimension']
    pp = PrettyPrinter()
    new_sequences, orig_indexes = memm.decrease_dim(evidences['sequences'], dim)
    new_dim = len(orig_indexes)
    str_sequences = [[(array_to_str(obs), state) for obs, state in seq] for seq in new_sequences]
    parents = list(graph.predecessors(user_id))
    parent_user_ids = [parents[index] for index in orig_indexes]

    print('Lambda:')
    pp.pprint(list(memm.Lambda))
    print('\nActivation probability of observations:')
    all_obs, _ = memm.get_all_obs_mat(new_sequences)
    print(f"{'decreased vector':{new_dim * 3}}{'probability'}")
    for i in range(all_obs.shape[0]):
        obs = all_obs[i, :]
        prob = memm.get_prob(obs)
        print(f"{array_to_str(obs):{new_dim * 3}}{prob:10.5f}")
    print('\nSelected indexes:')
    pp.pprint(orig_indexes)
    print('\nUser ids of selected indexes:')
    pp.pprint(parent_user_ids)
    print('\nEvidences with decreased dimensions:')
    pp.pprint(str_sequences)
    # print('\nEvidences:')
    # pp.pprint([[(obs_to_str(obs, dim), state) for obs, state in seq] for seq in evidences['sequences']])


def handle():
    parser = argparse.ArgumentParser('Display evidences of a user id')
    parser.add_argument('-p', '--project', type=str, help='project name', required=True)
    parser.add_argument('-u', '--userid', dest='user_id', type=str, help='user id', required=True)
    parser.add_argument('-m', '--method', type=str, help='MEMM method', required=True)
    args = parser.parse_args()

    methods = {e.value: e for e in MEMMMethod}
    project = Project(args.project)
    memm = MEMMManager(project, methods[args.method]).fetch_one(args.user_id)
    m = EvidenceManager(project, methods[args.method])
    evidences = m.get_one(ObjectId(args.user_id))
    graph = project.load_or_extract_graph()
    print_info(ObjectId(args.user_id), evidences, memm, graph)


if __name__ == '__main__':
    handle()
