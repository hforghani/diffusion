import argparse

from cascade.models import Project, ParamTypes
from memm.models import MEMM_EVID_FILE_NAME
from settings import mongodb, logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Insert MEMM evidences of a project into DB')
    parser.add_argument("-p", "--project", type=str, dest="project",
                        help="project name or multiple comma-separated project names")
    args = parser.parse_args()
    project = Project(args.project)
    logger.info('loading evidences ...')
    evidences = project.load_param(MEMM_EVID_FILE_NAME, ParamTypes.JSON)
    logger.info('preparing documents ...')
    evidences = [{'user_id': uid, 'dimension': value[0], 'evidences': value[1:]}
                 for uid, value in evidences.items()]
    logger.info('inserting documents ...')
    mongodb.memm_evid.insert_many(evidences)
