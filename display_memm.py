import argparse
from functools import reduce
from pprint import PrettyPrinter

from bson import ObjectId

from db.managers import EvidenceManager, DBManager, MEMMManager
from cascade.models import Project
from memm.enum import MEMMMethod
from memm.memm import MEMM, obs_to_str


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)

def print_info(user_id, evidences, memm, db_name):
    dim = evidences['dimension']
    pp = PrettyPrinter()
    sequences, orig_indexes = MEMM().decrease_dim(evidences['sequences'], dim)
    new_dim = len(orig_indexes)
    str_sequences = [[(obs_to_str(obs, new_dim), state) for obs, state in seq] for seq in sequences]
    manager = DBManager(db_name)
    parents = manager.db.relations.find_one({'user_id': ObjectId(user_id)}, {'parents': 1})['parents']
    parent_user_ids = [parents[index] for index in orig_indexes]

    print('Lambda:')
    pp.pprint(list(memm.Lambda))
    print('\nActivation probability of observations:')
    all_obs = list(reduce(lambda s1, s2: s1 | s2, [{pair[0] for pair in seq} for seq in evidences['sequences']]))
    print(f"{'decreased int':20}{'decreased vector':{new_dim + 5}}{'probability'}")
    for obs in all_obs:
        new_obs = MEMM.decrease_dim_by_indexes(obs, memm.orig_indexes)
        prob = memm.get_prob(obs)
        print(f"{new_obs:<20}{obs_to_str(new_obs, new_dim):{new_dim + 5}}{prob:10.5f}")
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
    args = parser.parse_args()

    method = MEMMMethod.BIN_MEMM
    project = Project(args.project)
    memm = MEMMManager(project, method).fetch_one(args.user_id)
    m = EvidenceManager(project, method)
    evidences = m.get_one(args.user_id)
    print_info(args.user_id, evidences, memm, project.db)


if __name__ == '__main__':
    handle()
