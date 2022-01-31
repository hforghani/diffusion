import gridfs
import pymongo
from bson import ObjectId
import numpy as np

from db.exceptions import DataDoesNotExist
from memm.memm import BinMEMM, TDMEMM, ParentTDMEMM, LongParentTDMEMM
from diffusion.enum import Method
from settings import logger, MONGO_URL


class DBManager:
    def __init__(self, db_name):
        mongo_client = pymongo.MongoClient(MONGO_URL)
        self.db = mongo_client[db_name]


class EvidenceManager:
    def __init__(self, project, method):
        self.project = project
        self.method = method
        mongo_client = pymongo.MongoClient(MONGO_URL)
        self.db = mongo_client[f'{project.db}_{method.value}_evid_{project.name}']

    def get_one(self, user_id):
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        fs = gridfs.GridFS(self.db)
        doc = fs.find_one({'user_id': user_id})
        if doc is None:
            raise ValueError(f'No evidence exists for user id {user_id}')
        return {
            'dimension': doc.dimension,
            'sequences': self._str_to_sequences(doc.read())
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
                    'sequences': self._str_to_sequences(doc.read())
                }
                for doc in documents
            }
        else:
            raise DataDoesNotExist(
                f'No MEMM evidences exist on project {self.project.name}'
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
                    'sequences': self._str_to_sequences(doc.read())
                }
                yield doc['user_id'], evidences
        else:
            raise DataDoesNotExist(
                f'No MEMM evidences exist on project {self.project.name}'
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
            fs.put(bytes(self._sequences_to_str(evidences[uid]['sequences']), encoding='utf8'),
                   user_id=ObjectId(uid),
                   dimension=evidences[uid]['dimension'])
            i += 1
            if i % 10000 == 0:
                logger.info('%d documents inserted', i)

    def _sequences_to_str(self, sequences):
        if self.method in [Method.BIN_MEMM, Method.REDUCED_BIN_MEMM]:
            return str([[(obs.astype(int).tolist(), state) for obs, state in seq] for seq in sequences])
        else:
            return str([[(obs.tolist(), state) for obs, state in seq] for seq in sequences])

    def _str_to_sequences(self, seq_str):
        if self.method in [Method.BIN_MEMM, Method.REDUCED_BIN_MEMM]:
            return [[(np.fromiter(obs, bool), state) for obs, state in seq] for seq in eval(seq_str)]
        else:
            return [[(np.fromiter(obs, np.float64), state) for obs, state in seq] for seq in eval(seq_str)]

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
    def __init__(self, project, method):
        self.project = project
        self.client = pymongo.MongoClient(MONGO_URL)
        self.db_name = f'{project.db}_{method.value}_{project.name}'
        self.db = self.client[self.db_name]
        self.method = method

    def db_exists(self):
        db_names = self.client.list_database_names()
        return self.db_name in db_names

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
        i = 0
        for doc in fs.find():
            memm = self.__doc_to_memm(doc)
            memms[doc.user_id] = memm
            i += 1
            if i % 10000 == 0:
                logger.debug('%d MEMMs fetched', i)
        return memms

    def fetch_one(self, user_id):
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        fs = gridfs.GridFS(self.db)
        doc = fs.find_one({'user_id': user_id})
        if doc is None:
            return None
        memm = self.__doc_to_memm(doc)
        return memm

    def __get_doc(self, memm):
        doc = {
            'orig_indexes': memm.orig_indexes,
            'lambda': memm.Lambda.tolist()
        }
        return doc

    def __doc_to_memm(self, doc):
        data = doc.read()
        memm_data = eval(data)
        if self.method in [Method.BIN_MEMM, Method.REDUCED_BIN_MEMM]:
            memm = BinMEMM()
        elif self.method == Method.PARENT_SENS_TD_MEMM:
            memm = ParentTDMEMM()
        elif self.method == Method.LONG_PARENT_SENS_TD_MEMM:
            memm = LongParentTDMEMM()
        else:
            memm = TDMEMM()
        memm.orig_indexes = memm_data['orig_indexes']
        memm.Lambda = np.fromiter(memm_data['lambda'], np.float64)
        return memm
