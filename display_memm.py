import argparse
from pprint import PrettyPrinter

from bson import ObjectId

from db.managers import EvidenceManager, DBManager, MEMMManager
from cascade.models import Project
from memm.memm import MEMM, array_to_obs

# import pydevd_pycharm
#
# pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Display evidences of a user id')
    parser.add_argument('-p', '--project', type=str, help='project name', required=True)
    parser.add_argument('-u', '--userid', dest='user_id', type=str, help='user id', required=True)
    args = parser.parse_args()

    project = Project(args.project)
    m = EvidenceManager(project)
    evidences = m.get_one(args.user_id)
    dim = evidences['dimension']
    pp = PrettyPrinter()
    sequences, orig_indexes = MEMM().decrease_dim(evidences['sequences'], dim)
    sequences = [[(bin(int(obs))[2:], state) for obs, state in seq] for seq in sequences]
    manager = DBManager()
    parents = manager.db.relations.find_one({'user_id': ObjectId(args.user_id)}, {'parents': 1})['parents']
    parent_user_ids = [parents[index] for index in orig_indexes]

    memm = MEMMManager(project).fetch_one(args.user_id)
    c1_width = memm.all_obs_arr.shape[1] * 2 + 10
    print('TPM:')
    print("{:<{w}}{:<30}{:<30}".format(' ', 0, 1, w=c1_width))
    for i in range(memm.all_obs_arr.shape[0]):
        obs = bin(array_to_obs(memm.all_obs_arr[i, :]))[2:]
        print("{:<{w}}{:<30}{:<30}".format(obs, memm.TPM[i, 0], memm.TPM[i, 1], w=c1_width))

    print('\nSelected indexes:')
    pp.pprint(orig_indexes)
    print('\nUser ids of selected indexes:')
    pp.pprint(parent_user_ids)
    print('\nEvidences with decreased dimensions:')
    pp.pprint(sequences)
