import pickle

import pymongo
from bson import ObjectId, Binary
import numpy as np

from db.exceptions import DataDoesNotExist
from memm.memm import MEMM
from settings import logger, MONGO_URL, DB_NAME


class DBManager:
    def __init__(self):
        mongo_client = pymongo.MongoClient(MONGO_URL)
        self.db = mongo_client[DB_NAME]


class EvidenceManager:
    def __init__(self):
        mongo_client = pymongo.MongoClient(MONGO_URL)
        self.db = mongo_client[DB_NAME]

    def _get_collection(self, project):
        return self.db.get_collection(f'memm_evid_{project.project_name}')

    def get(self, project, user_id):
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        collection = self._get_collection(project)
        docs = collection.find({'user_id': user_id})
        dim = docs[0]['dimension']
        docs.rewind()
        evidences = {
            'dimension': dim,
            'sequences': [
                [
                    (int(obs_state[0]), obs_state[1]) for obs_state in doc['sequence']
                ] for doc in docs
            ]
        }
        return evidences

    def __find_by_user_ids(self, project, user_ids):
        collection = self._get_collection(project)

        if user_ids:
            user_ids = [uid if isinstance(uid, ObjectId) else ObjectId(uid) for uid in user_ids]

            documents = collection.aggregate([
                {'$match': {'user_id': {'$in': user_ids}}},
                {'$group': {
                    '_id': {'user_id': '$user_id', 'dimension': '$dimension'},
                    'sequences': {'$push': '$sequence'}
                }}
            ])
        else:
            documents = collection.aggregate([
                {'$group': {
                    '_id': {'user_id': '$user_id', 'dimension': '$dimension'},
                    'sequences': {'$push': '$sequence'}
                }}
            ])
        return documents

    def get_many(self, project, user_ids=None):
        """
        Return dictionary of user id's to the dict {'dimension': dim, 'sequences': sequences}
        of which sequences is the list of the sequences and each sequence is the list of (obs, state)
        tuples.
        :param project:
        :param user_ids:
        :return:
        """
        documents = self.__find_by_user_ids(project, user_ids)

        return {
            doc['user_id']: {
                'dimension': doc['dimension'],
                'sequences': [
                    [
                        (int(obs_state[0]), obs_state[1]) for obs_state in seq
                    ] for seq in doc['sequences']
                ]
            }
            for doc in documents
        }

    def get_many_generator(self, project, user_ids=None):
        """
        Get the generator of (user_id, evidences) tuples which each evidences is a dictionary
        {'dimension': dim, 'sequences': sequences}. Read the doc of get_many for more information.
        :param project:
        :param user_ids:
        :return:
        """
        documents = self.__find_by_user_ids(project, user_ids)

        if documents.count():
            for doc in documents:
                evidences = {
                    'dimension': doc['dimension'],
                    'sequences': [
                        [
                            (int(obs_state[0]), obs_state[1]) for obs_state in seq
                        ] for seq in doc['sequences']
                    ]
                }
                yield doc['user_id'], evidences
        else:
            raise DataDoesNotExist(
                f'No MEMM evidences exist on project {project.project_name}'
                f'{" for user set given" if user_ids else ""}')

    def insert(self, project, evidences):
        """
        :param project:
        :param evidences: dictionary of user id's to MEMM evidences. Each evidence is a dictionary
         with 2 keys:
            <p>dimension : number of observation dimensions.</p>
            <p>sequences : list of (obs, state) sequences.</p>
        :return:
        """
        docs = []
        for uid, evid in evidences.items():
            dim = evid['dimension']
            docs.extend([{
                'user_id': ObjectId(uid),
                'dimension': dim,
                'sequence': [[str(obs_state[0]), obs_state[1]] for obs_state in seq]
            } for seq in evid['sequences']])

        logger.info('inserting %d documents for %d users ...', len(docs), len(evidences))
        collection = self._get_collection(project)
        collection.insert_many(docs)

    def create_index(self, project):
        """
        Create index on 'user_id' key of MEMM evidences collection of the given project if does not exist.
        :param project:
        :return:
        """
        collection = self._get_collection(project)
        has_index = False
        for _, value in collection.index_information():
            if value['key'][0] == 'user_id':
                has_index = True
        if not has_index:
            collection.create_index('user_id')


class MEMMManager:
    def __init__(self):
        mongo_client = pymongo.MongoClient(MONGO_URL)
        self.db = mongo_client[DB_NAME]

    def __get_doc(self, user_id, memm):
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

    def insert(self, project, memms):
        logger.debug('creating MEMM documents ...')
        documents = [self.__get_doc(uid, memms[uid]) for uid in memms]
        logger.debug('inserting MEMMs into db ...')
        collection = self.db.get_collection(f'memm_{project.project_name}')
        collection.insert_many(documents)

    def fetch(self, project):
        collection = self.db.get_collection(f'memm_{project.project_name}')
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
