import argparse

from cascade.models import Project, ParamTypes
from db.managers import EvidenceManager
from memm.models import MEMM_EVID_FILE_NAME
from settings import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Insert MEMM evidences of a project into DB')
    parser.add_argument("-p", "--project", type=str, dest="project",
                        help="project name or multiple comma-separated project names")
    args = parser.parse_args()
    project = Project(args.project)
    logger.info('loading evidences ...')
    evidences = project.load_param(MEMM_EVID_FILE_NAME, ParamTypes.JSON)
    logger.info('preparing documents ...')
    evidences = {uid: {
        'dimension': value[0],
        'sequences': [[[str(obs_state[0]), obs_state[1]] for obs_state in seq] for seq in value[1]]
    } for uid, value in evidences.items()}

    manager = EvidenceManager()
    manager.insert(project, evidences)