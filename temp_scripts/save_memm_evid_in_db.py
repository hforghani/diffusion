import argparse

from cascade.models import Project, ParamTypes
from db.managers import EvidenceManager
from memm.models import MEMM_EVID_FILE_NAME
from settings import logger
from utils.time_utils import Timer

if __name__ == '__main__':
    with Timer('save_memm_evid_in_db'):
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
            'sequences': value[1]
        } for uid, value in evidences.items()}

        manager = EvidenceManager()
        manager.insert(project, evidences)