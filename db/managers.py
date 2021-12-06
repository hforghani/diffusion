import pickle

import gridfs
import pymongo
from bson import ObjectId, Binary
import numpy as np
from scipy.sparse import csr_matrix

from db.exceptions import DataDoesNotExist
from memm.memm import MEMM
from settings import logger, MONGO_URL, DB_NAME


class DBManager:
    def __init__(self):
        mongo_client = pymongo.MongoClient(MONGO_URL)
        self.db = mongo_client[DB_NAME]


class EvidenceManager:
    def __init__(self, project):
        self.project = project
        mongo_client = pymongo.MongoClient(MONGO_URL)
        self.db = mongo_client[f'{DB_NAME}_memm_evid_{project.project_name}']

    def get_one(self, user_id):
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        fs = gridfs.GridFS(self.db)
        docs = fs.find({'user_id': user_id})
        if not docs:
            raise ValueError('No evidence exists for user id %s', user_id)
        return {
            'dimension': docs[0].dimension,
            'sequences': eval(docs[0].read())
        }

    def __find_by_user_ids(self, user_ids):
        fs = gridfs.GridFS(self.db)

        if user_ids:
            documents = fs.find({'user_id': {'$in': user_ids}}, no_cursor_timeout=True)
        else:
            documents = fs.find(no_cursor_timeout=True)

        return documents

    def get_many(self, user_ids=None):
        """
        Return dictionary of user id's to the dict {'dimension': dim, 'sequences': sequences}
        of which sequences is the list of the sequences and each sequence is the list of (obs, state)
        tuples.
        :param user_ids:
        :return:
        """
        documents = self.__find_by_user_ids(user_ids)

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
                f'No MEMM evidences exist on project {self.project.project_name}'
                f'{" for user set given" if user_ids else ""}')

    def get_many_generator(self, user_ids=None):
        """
        Get the generator of (user_id, evidences) tuples which each evidences is a dictionary
        {'dimension': dim, 'sequences': sequences}. Read the doc of get_many for more information.
        :param user_ids:
        :return:
        """
        documents = self.__find_by_user_ids(user_ids)

        if documents.count():
            for doc in documents:
                evidences = {
                    'dimension': doc.dimension,
                    'sequences': eval(doc.read())
                }
                yield doc['user_id'], evidences
        else:
            raise DataDoesNotExist(
                f'No MEMM evidences exist on project {self.project.project_name}'
                f'{" for user set given" if user_ids else ""}')

    def insert(self, evidences):
        """
        :param evidences: dictionary of user id's to MEMM evidences. Each evidence is a dictionary
         with 2 keys:
            <p>dimension : number of observation dimensions.</p>
            <p>sequences : list of (obs, state) sequences.</p>
        :return:
        """
        fs = gridfs.GridFS(self.db)

        logger.info('inserting %d MEMM evidence documents ...', len(evidences))
        i = 0
        for uid in evidences:
            fs.put(bytes(str(evidences[uid]['sequences']), encoding='utf8'),
                   user_id=ObjectId(uid),
                   dimension=evidences[uid]['dimension'])
            i += 1
            if i % 10000 == 0:
                logger.info('%d documents inserted', i)

    def create_index(self):
        """
        Create index on 'user_id' key of MEMM evidences collection of the given project if does not exist.
        :return:
        """
        collection = self.db.get_collection('fs.files')
        for _, value in collection.index_information().items():
            if value['key'][0][0] == 'user_id':
                break
        else:
            collection.create_index('user_id')


class MEMMManager:
    def __init__(self, project):
        self.project = project
        mongo_client = pymongo.MongoClient(MONGO_URL)
        self.db = mongo_client[f'{DB_NAME}_memm_{project.project_name}']

    def insert(self, memms):
        fs = gridfs.GridFS(self.db)

        logger.debug('inserting %d MEMM documents ...', len(memms))
        i = 0
        for uid in memms:
            doc = self.__get_doc(memms[uid])
            fs.put(bytes(str(doc), encoding='utf8'), user_id=uid)
            i += 1
            if i % 10000 == 0:
                logger.debug('%d documents inserted', i)

    def fetch_all(self):
        fs = gridfs.GridFS(self.db)
        memms = {}

        for doc in fs.find():
            memm = self.__doc_to_memm(doc)
            memms[doc.user_id] = memm
        return memms

    def fetch_one(self, user_id):
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        fs = gridfs.GridFS(self.db)
        doc = fs.find_one({'user_id': user_id})
        memm = self.__doc_to_memm(doc)
        return memm

    def __get_doc(self, memm):
        doc = {
            'lambda': memm.Lambda.tolist(),
            'tpm': pickle.dumps(memm.TPM, protocol=2),
            'all_obs_arr': pickle.dumps(csr_matrix(memm.all_obs_arr), protocol=2),
            'map_obs_index': {str(key): value for key, value in memm.map_obs_index.items()},
            'orig_indexes': memm.orig_indexes
        }
        if isinstance(doc['orig_indexes'], dict):
            doc['orig_indexes'] = sorted(list(doc['orig_indexes'].values()))
        return doc

    def __doc_to_memm(self, doc):
        data = doc.read()
        try:
            memm_data = eval(data)
        except OverflowError:
            logger.debug('eval function does not work for a MEMM data with length %d. Will be divided.', doc.length)
            memm_data = self.__parse_doc(data)
        memm = MEMM()
        memm.Lambda = np.fromiter(memm_data['lambda'], np.float64)
        memm.TPM = pickle.loads(memm_data['tpm'])
        memm.all_obs_arr = pickle.loads(memm_data['all_obs_arr']).toarray()
        memm.map_obs_index = {int(key): value for key, value in memm_data['map_obs_index'].items()}
        memm.orig_indexes = memm_data['orig_indexes']
        return memm

    def __parse_doc(self, data):
        memm_data = {}
        i1 = data.index(b'lambda')
        i2 = data.index(b'tpm')
        i3 = data.index(b'all_obs_arr')
        i4 = data.index(b'map_obs_index')

        # print('tpm binary dump =', data[i2 + 15: i3 - 8])
        # print('all_obs_arr dump =', data[i3 + 23: i4 - 8])
        memm_data['lambda'] = eval(data[i1 + 9: i2 - 3])
        memm_data['tpm'] = eval(data[i2 + 15: i3 - 8])
        memm_data['all_obs_arr'] = eval(data[i3 + 23: i4 - 8])
        two_last_keys = eval(b"{" + data[i4 - 1:])
        memm_data.update(two_last_keys)
        # logger.debug('saved memm: %s', memm_data)

        return memm_data
