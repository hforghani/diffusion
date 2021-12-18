import argparse
from pprint import PrettyPrinter

from bson import ObjectId

from db.managers import EvidenceManager, DBManager, MEMMManager
from cascade.models import Project
from memm.memm import MEMM, array_to_str, obs_to_str


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)

def print_info(user_id, evidences, memm):
    dim = evidences['dimension']
    pp = PrettyPrinter()
    sequences, orig_indexes = MEMM().decrease_dim(evidences['sequences'], dim)
    new_dim = len(orig_indexes)
    sequences = [[(obs_to_str(obs, new_dim), state) for obs, state in seq] for seq in sequences]
    manager = DBManager()
    parents = manager.db.relations.find_one({'user_id': ObjectId(user_id)}, {'parents': 1})['parents']
    parent_user_ids = [parents[index] for index in orig_indexes]

    print('Activation probability of observations:')
    pp.pprint({obs_to_str(obs, new_dim): prob for obs, prob in memm.map_obs_prob.items()})
    print('\nSelected indexes:')
    pp.pprint(orig_indexes)
    print('\nUser ids of selected indexes:')
    pp.pprint(parent_user_ids)
    print('\nEvidences with decreased dimensions:')
    pp.pprint(sequences)
    print('\nEvidences:')
    pp.pprint([[(obs_to_str(obs, dim), state) for obs, state in seq] for seq in evidences['sequences']])


def handle():
    parser = argparse.ArgumentParser('Display evidences of a user id')
    parser.add_argument('-p', '--project', type=str, help='project name', required=True)
    parser.add_argument('-u', '--userid', dest='user_id', type=str, help='user id', required=True)
    args = parser.parse_args()

    project = Project(args.project)
    memm = MEMMManager(project).fetch_one(args.user_id)
    m = EvidenceManager(project)
    evidences = m.get_one(args.user_id)
    print_info(args.user_id, evidences, memm)


if __name__ == '__main__':
    handle()
