import pickle

import gridfs
import pymongo
from bson import ObjectId

from db.exceptions import DataDoesNotExist
from seq_labeling.pgm import *
from diffusion.enum import Method
from settings import logger, MONGO_URL


class DBManager:
    def __init__(self, db_name):
        self.client = pymongo.MongoClient(MONGO_URL)
        self.db = self.client[db_name]


class EvidenceManager:
    def __init__(self, project):
        self.project = project
        mongo_client = pymongo.MongoClient(MONGO_URL)
        db_name = self.get_db_name(project)
        self.db = mongo_client[db_name]

    def get_db_name(self, project):
        return f'{project.db}_evid_{project.name}'

    def get_one(self, user_id):
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        fs = gridfs.GridFS(self.db)
        doc = fs.find_one({'user_id': user_id})
        if doc is None:
            raise ValueError(f'No evidence exists for user id {user_id}')
        return self._str_to_sequences(doc.read())

    def __find_by_user_ids(self, user_ids):
        fs = gridfs.GridFS(self.db)

        if user_ids:
            documents = fs.find({'user_id': {'$in': user_ids}}, no_cursor_timeout=True)
        else:
            documents = fs.find(no_cursor_timeout=True)

        return documents

    def get_many(self, user_ids=None):
        """
        Return dictionary of user id's to the lists of the sequences. Each sequence is a list of (obs, state) tuples.
        :param user_ids:
        :return:
        """
        documents = self.__find_by_user_ids(user_ids)

        if documents.count():
            return {
                doc.user_id: self._str_to_sequences(doc.read()) for doc in documents
            }
        else:
            raise DataDoesNotExist(
                f'No MEMM evidences exist on project {self.project.name}'
                f'{" for user set given" if user_ids else ""}')

    def get_many_generator(self, user_ids=None):
        """
        Get the generator of (user_id, sequences) tuples which each "sequences" is a list of (obs, state) tuples.
        :param user_ids:
        :return:
        """
        documents = self.__find_by_user_ids(user_ids)

        if documents.count():
            for doc in documents:
                yield doc['user_id'], self._str_to_sequences(doc.read())
        else:
            raise DataDoesNotExist(
                f'No MEMM evidences exist on project {self.project.name}'
                f'{" for user set given" if user_ids else ""}')

    def insert(self, evidences):
        """
        :param evidences: dictionary of user id's to the sequences.
        :return:
        """
        fs = gridfs.GridFS(self.db)

        logger.info('inserting %d evidence documents ...', len(evidences))
        i = 0
        for uid in evidences:
            fs.put(bytes(self._sequences_to_str(evidences[uid]), encoding='utf8'),
                   user_id=ObjectId(uid))
            i += 1
            if i % 10000 == 0:
                logger.info('%d documents inserted', i)

    def _sequences_to_str(self, sequences):
        return str([[(obs.astype(int).tolist(), state) for obs, state in seq] for seq in sequences])

    def _str_to_sequences(self, seq_str):
        return [[(np.array(obs, dtype=bool), state) for obs, state in seq] for seq in eval(seq_str)]

    def create_index(self):
        """
        Create index on 'user_id' key of the collection of evidences of the given project if it does not exist.
        :return:
        """
        collection = self.db.get_collection('fs.files')
        for _, value in collection.index_information().items():
            if value['key'][0][0] == 'user_id':
                break
        else:
            collection.create_index('user_id')


class ParentSensEvidManager(EvidenceManager):
    def get_db_name(self, project):
        return f'{project.db}_parent_evid_{project.name}'


class SeqLabelDBManager:
    def __init__(self, project, method):
        self.project = project
        self.client = pymongo.MongoClient(MONGO_URL)
        self.db_name = f'{project.db}_{method.value}_{project.name}'
        self.db = self.client[self.db_name]
        self.method = method

    def db_exists(self):
        db_names = self.client.list_database_names()
        return self.db_name in db_names

    def insert(self, models):
        fs = gridfs.GridFS(self.db)

        logger.debug('inserting %d model documents ...', len(models))
        i = 0
        for uid in models:
            doc = self._get_doc(models[uid])
            fs.put(bytes(str(doc), encoding='utf8'), user_id=uid)
            i += 1
            if i % 10000 == 0:
                logger.debug('%d documents inserted', i)

    def fetch_all(self):
        fs = gridfs.GridFS(self.db)
        models = {}
        i = 0
        for doc in fs.find():
            memm = self._doc_to_model(doc)
            models[doc.user_id] = memm
            i += 1
            if i % 10000 == 0:
                logger.debug('%d models fetched', i)
        return models

    def fetch_one(self, user_id):
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        fs = gridfs.GridFS(self.db)
        doc = fs.find_one({'user_id': user_id})
        if doc is None:
            return None
        model = self._doc_to_model(doc)
        return model

    def _get_doc(self, model):
        doc = {
            'orig_indexes': model.orig_indexes,
            'lambda': model.Lambda.tolist(),
        }
        if hasattr(model, 'td_param'):
            doc['td_param'] = model.td_param
        return doc

    def _doc_to_model(self, doc):
        data = doc.read()
        model_data = eval(data)
        if self.method in [Method.LONG_MEMM, Method.MULTI_STATE_LONG_MEMM]:
            model = LongMEMM()
        elif self.method in [Method.BIN_MEMM, Method.MULTI_STATE_BIN_MEMM]:
            model = BinMEMM()
        elif self.method == Method.PARENT_SENS_TD_MEMM:
            model = ParentTDMEMM()
        elif self.method == Method.LONG_PARENT_SENS_TD_MEMM:
            model = LongParentTDMEMM()
        else:
            model = TDMEMM()
        orig_indexes = model_data['orig_indexes']
        Lambda = np.fromiter(model_data['lambda'], np.float64)
        model.set_params(Lambda, orig_indexes)
        if 'td_param' in model_data:
            model.td_param = model_data['td_param']
        return model


class CRFManager(SeqLabelDBManager):
    def db_exists(self):
        return False  # TODO: This is a temporary hack to prevent reading from and inserting into db for CRF models.

    def _get_doc(self, crf):
        return pickle.dumps(crf)

    def _doc_to_model(self, doc):
        data = doc.read()
        crf = pickle.loads(data)
        return crf
