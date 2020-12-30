import pickle

import pymongo
from bson import ObjectId, Binary, InvalidDocument
import numpy as np

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
            'lambda': memm.Lambda.tolist(),
            'tpm': Binary(pickle.dumps(memm.TPM, protocol=2)),
            'all_obs_arr': Binary(pickle.dumps(memm.all_obs_arr, protocol=2)),
            'map_obs_index': {str(key): value for key, value in memm.map_obs_index.items()},
            'orig_indexes': memm.orig_indexes
        }

    @staticmethod
    def insert(project, memms):
        logger.debug('creating MEMM documents ...')
        documents = [MEMMManager.__get_doc(uid, memms[uid]) for uid in memms]
        logger.debug('inserting MEMMs into db ...')
        try:
            mongodb.memms.insert_one({
                'project_name': project.project_name,
                'memms': documents
            })
        except InvalidDocument:
            for i in range(10):
                logger.debug(documents[i])
            raise

    @staticmethod
    def fetch(project):
        memms_data = mongodb.memms.find_one({'project_name': project.project_name},
                                            {'memms': 1, '_id': 0})
        if memms_data is None:
            return {}

        memms = {}
        for doc in memms_data['memms']:
            memm = MEMM()
            memm.Lambda = np.fromiter(doc['lambda'], np.float64)
            memm.TPM = pickle.loads(doc['tpm'])
            memm.all_obs_arr = pickle.loads(doc['all_obs_arr'])
            memm.map_obs_index = {int(key): value for key, value in doc['map_obs_index'].items()}
            memm.orig_indexes = doc['orig_indexes']
            memms[doc['user_id']] = memm
        return memms
