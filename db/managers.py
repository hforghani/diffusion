import pickle

import gridfs
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
        return self.db.get_collection(f'{DB_NAME}_memm_evid_{project.project_name}')

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
        mongo_client = pymongo.MongoClient(MONGO_URL)
        evid_db = mongo_client[f'{DB_NAME}_memm_evid_{project.project_name}']
        fs = gridfs.GridFS(evid_db)

        if user_ids:
            documents = fs.find({'user_id': {'$in': user_ids}}, no_cursor_timeout=True)
        else:
            documents = fs.find(no_cursor_timeout=True)

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

        if documents.count():
            return {
                doc.user_id: {
                    'dimension': doc.dimension,
                    'sequences': eval(doc.read())
                }
                for doc in documents
            }
        else:
            raise DataDoesNotExist(
                f'No MEMM evidences exist on project {project.project_name}'
                f'{" for user set given" if user_ids else ""}')

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
                    'dimension': doc.dimension,
                    'sequences': eval(doc.read())
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
        mongo_client = pymongo.MongoClient(MONGO_URL)
        evid_db = mongo_client[f'{DB_NAME}_memm_evid_{project.project_name}']
        fs = gridfs.GridFS(evid_db)

        logger.info('inserting %d MEMM evidence documents ...', len(evidences))
        i = 0
        for uid in evidences:
            fs.put(bytes(str(evidences[uid]['sequences']), encoding='utf8'),
                   user_id=ObjectId(uid),
                   dimension=evidences[uid]['dimension'])
            i += 1
            if i % 10000 == 0:
                logger.info('%d documents inserted', i)

    def create_index(self, project):
        """
        Create index on 'user_id' key of MEMM evidences collection of the given project if does not exist.
        :param project:
        :return:
        """
        collection = self._get_collection(project)
        for _, value in collection.index_information():
            if value['key'][0] == 'user_id':
                break
        else:
            collection.create_index('user_id')


class MEMMManager:
    # def __init__(self):
    #     mongo_client = pymongo.MongoClient(MONGO_URL)
    #     self.db = mongo_client[DB_NAME]

    @staticmethod
    def __get_doc(memm):
        doc = {
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
        # logger.debug('creating MEMM documents ...')
        # documents = [self.__get_doc(uid, memms[uid]) for uid in memms]
        # logger.debug('inserting MEMMs into db ...')
        # collection = self.db.get_collection(f'memm_{project.project_name}')
        # collection.insert_many(documents)

        mongo_client = pymongo.MongoClient(MONGO_URL)
        memm_db = mongo_client[f'{DB_NAME}_memm_{project.project_name}']
        fs = gridfs.GridFS(memm_db)

        logger.info('inserting %d MEMM documents ...', len(memms))
        i = 0
        for uid in memms:
            doc = MEMMManager.__get_doc(memms[uid])
            fs.put(bytes(str(doc), encoding='utf8'), user_id=uid)
            i += 1
            if i % 10000 == 0:
                logger.info('%d documents inserted', i)

    @staticmethod
    def fetch(project):
        # collection = self.db.get_collection(f'memm_{project.project_name}')
        # memms = {}
        # for doc in collection.find():
        #     memm = MEMM()
        #     memm.Lambda = np.fromiter(doc['lambda'], np.float64)
        #     memm.TPM = pickle.loads(doc['tpm'])
        #     memm.all_obs_arr = pickle.loads(doc['all_obs_arr'])
        #     memm.map_obs_index = {int(key): value for key, value in doc['map_obs_index'].items()}
        #     memm.orig_indexes = doc['orig_indexes']
        #     memms[doc['user_id']] = memm
        # return memms

        mongo_client = pymongo.MongoClient(MONGO_URL)
        memm_db = mongo_client[f'{DB_NAME}_memm_{project.project_name}']
        fs = gridfs.GridFS(memm_db)
        memms = {}
        for doc in fs.find():
            memm_data = eval(doc.read())
            memm = MEMM()
            memm.Lambda = np.fromiter(memm_data['lambda'], np.float64)
            memm.TPM = pickle.loads(memm_data['tpm'])
            memm.all_obs_arr = pickle.loads(memm_data['all_obs_arr'])
            memm.map_obs_index = {int(key): value for key, value in memm_data['map_obs_index'].items()}
            memm.orig_indexes = memm_data['orig_indexes']
            memms[doc.user_id] = memm
        return memms
