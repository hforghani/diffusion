import pickle

import pymongo
from bson import ObjectId, Binary, InvalidDocument
import numpy as np

from memm.memm import MEMM
from settings import mongodb, logger


class EvidenceManager:
    @staticmethod
    def get(project, user_id):
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        collection = mongodb.get_collection(f'memm_evid_{project.project_name}')
        doc = collection.find_one({'user_id': user_id})
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

    @staticmethod
    def get_many(project, user_ids):
        user_ids = [uid if isinstance(uid, ObjectId) else ObjectId(uid) for uid in user_ids]

        collection = mongodb.get_collection(f'memm_evid_{project.project_name}')
        documents = collection.find({'user_id': {'$in': user_ids}})
        for doc in documents:
            evidences = [
                doc['dimension'],
                [
                    [
                        [int(obs_state[0]), obs_state[1]] for obs_state in seq
                    ] for seq in doc['evidences']
                ]
            ]
            yield doc['user_id'], evidences


class MEMMManager:
    @staticmethod
    def __get_doc(user_id, memm):
        doc = {
            'user_id': user_id,
            'lambda': memm.Lambda.tolist(),
            'tpm': Binary(pickle.dumps(memm.TPM, protocol=2)),
            'all_obs_arr': Binary(pickle.dumps(memm.all_obs_arr, protocol=2)),
            'map_obs_index': {str(key): value for key, value in memm.map_obs_index.items()},
            'orig_indexes': memm.orig_indexes
        }
        if isinstance(doc['orig_indexes'], dict):
            doc['orig_indexes'] = sorted(list(doc['orig_indexes'].values()))
        return doc

    @staticmethod
    def insert(project, memms):
        logger.debug('creating MEMM documents ...')
        documents = [MEMMManager.__get_doc(uid, memms[uid]) for uid in memms]
        logger.debug('inserting MEMMs into db ...')
        collection = mongodb.get_collection(f'memms_{project.project_name}')
        collection.insert_many(documents)

    @staticmethod
    def fetch(project):
        collection = mongodb.get_collection(f'memms_{project.project_name}')
        memms = {}
        for doc in collection.find():
            memm = MEMM()
            memm.Lambda = np.fromiter(doc['lambda'], np.float64)
            memm.TPM = pickle.loads(doc['tpm'])
            memm.all_obs_arr = pickle.loads(doc['all_obs_arr'])
            memm.map_obs_index = {int(key): value for key, value in doc['map_obs_index'].items()}
            memm.orig_indexes = doc['orig_indexes']
            memms[doc['user_id']] = memm
        return memms
