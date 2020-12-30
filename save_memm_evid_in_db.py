import argparse

from bson import ObjectId

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
    evidences = [{'user_id': ObjectId(uid),
                  'dimension': value[0],
                  'evidences': [[[str(obs_state[0]), obs_state[1]] for obs_state in seq] for seq in value[1]]}
                 for uid, value in evidences.items()]

    logger.info('inserting documents ...')
    collection = mongodb.get_collection(f'memm_evid_{project.project_name}')
    collection.insert_many(evidences)
