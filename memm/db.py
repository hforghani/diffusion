import pickle

import pymongo
from bson import ObjectId

from memm.memm import MEMM
from settings import mongodb, logger


class EvidenceManager:
    @staticmethod
    def get(user_id):
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        doc = mongodb.memm_evid.find_one({'user_id': user_id})
        if doc is None:
            return None
        else:
            evidences = [
                doc['dimension'],
                [
                    [
                        [int(obs_state[0]), obs_state[1]] for obs_state in seq
                    ] for seq in doc['evidences']
                ]
            ]
            return evidences


class MEMMManager:
    @staticmethod
    def __get_doc(user_id, memm):
        return {
            'user_id': user_id,
            'lambda': memm.Lambda,
            'tpm': pymongo.binary.Binary(pickle.dumps(memm.TPM, protocol=2)),
            'all_obs_arr': pymongo.binary.Binary(pickle.dumps(memm.all_obs_arr, protocol=2)),
            'map_obs_index': memm.map_obs_index,
            'orig_indexes': memm.orig_indexes
        }

    @staticmethod
    def insert(project, memms):
        logger.debug('creating MEMM documents ...')
        documents = [MEMMManager.__get_doc(uid, memms[uid]) for uid in memms]
        logger.debug('inserting MEMMs into db ...')
        mongodb.memms.insert_one({
            'project_name': project.project_name,
            'memms': documents
        })

    @staticmethod
    def fetch(project):
        memms_data = mongodb.memms.find_one({'project_name': project.project_name},
                                            {'memms': 1, '_id': 0})
        if memms_data is None:
            return {}

        memms = {}
        for doc in memms_data['memms']:
            memm = MEMM()
            memm.Lambda = doc['lambda']
            memm.TPM = pickle.loads(doc['tpm'])
            memm.all_obs_arr = pickle.loads(doc['all_obs_arr'])
            memm.map_obs_index = doc['map_obs_index']
            memm.orig_indexes = doc['orig_indexes']
            memms[doc['user_id']] = memm
        return memms
