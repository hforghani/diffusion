import argparse

from cascade.models import Project
from db.managers import EvidenceManager
from display_memm import print_info
from memm.memm import MEMM

# import pydevd_pycharm
#
# pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Learn a MEMM for a user id and display it')
    parser.add_argument('-p', '--project', type=str, help='project name', required=True)
    parser.add_argument('-u', '--userid', dest='user_id', type=str, help='user id', required=True)
    args = parser.parse_args()

    project = Project(args.project)
    m = EvidenceManager(project)
    evidences = m.get_one(args.user_id)
    memm = MEMM()
    memm.fit(evidences)
    print_info(args.user_id, evidences, memm)
