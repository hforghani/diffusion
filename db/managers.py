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


class EvidenceManager(DBManager):
    def __init__(self, project):
        self.project = project
        db_name = self.get_db_name(project)
        super().__init__(db_name)
        self.train_sets_col = self.client['train_sets'][f'{project.db}_{project.name}']

    def get_db_name(self, project):
        return f'{project.db}_evid_{project.name}'

    def get_one(self, user_id, train_set):
        set_id = self.__get_train_set_id(train_set)
        if set_id is None:
            raise DataDoesNotExist('No evidences exist for training set given')
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        fs = gridfs.GridFS(self.db)
        doc = fs.find_one({'set_id': set_id, 'user_id': user_id})
        if doc is None:
            raise ValueError(f'No evidence exists for user id {user_id}')
        return self._str_to_sequences(doc.read())

    def __find_by_set_id(self, set_id):
        fs = gridfs.GridFS(self.db)
        return fs.find({'set_id': set_id}, no_cursor_timeout=True)

    def __get_train_set_id(self, train_set):
        set_id = None
        for doc in self.train_sets_col.find():
            if set(doc['train_set']) == set(train_set):
                set_id = doc['_id']
        return set_id

    def get_many(self, train_set):
        """
        Return dictionary of user id's to the lists of the sequences. Each sequence is a list of (obs, state) tuples.
        :param user_ids:
        :return:
        """
        set_id = self.__get_train_set_id(train_set)
        if set_id is None:
            raise DataDoesNotExist('No evidences exist for training set given')

        documents = self.__find_by_set_id(set_id)
        if documents.count():
            return {
                doc.user_id: self._str_to_sequences(doc.read()) for doc in documents
            }
        else:
            raise DataDoesNotExist(f'No evidences exist for training set given')

    def get_many_generator(self, train_set):
        """
        Get the generator of (user_id, sequences) tuples which each "sequences" is a list of (obs, state) tuples.
        :param user_ids:
        :return:
        """
        set_id = self.__get_train_set_id(train_set)
        if set_id is None:
            raise DataDoesNotExist('No evidences exist for training set given')

        documents = self.__find_by_set_id(set_id)
        if documents.count():
            for doc in documents:
                yield doc['user_id'], self._str_to_sequences(doc.read())
        else:
            raise DataDoesNotExist('No evidences exist for training set given')

    def insert(self, evidences, train_set):
        """
        :param evidences: dictionary of user id's to the sequences.
        :return:
        """
        set_id = self.__get_train_set_id(train_set)
        if set_id is None:
            set_id = self.train_sets_col.insert_one({'train_set': train_set}).inserted_id

        fs = gridfs.GridFS(self.db)
        logger.info('inserting %d evidence documents ...', len(evidences))
        i = 0
        for uid in evidences:
            fs.put(bytes(self._sequences_to_str(evidences[uid]), encoding='utf8'), user_id=uid, set_id=set_id)
            i += 1
            if i % 10000 == 0:
                logger.info('%d documents inserted', i)

    def _sequences_to_str(self, sequences):
        return str([[(obs.astype(int).tolist(), state) for obs, state in seq] for seq in sequences])

    def _str_to_sequences(self, seq_str):
        return [[(np.array(obs, dtype=bool), state) for obs, state in seq] for seq in eval(seq_str)]

    def create_index(self):
        """
        Create index on 'set_id' key of the collection if it does not exist.
        :return:
        """
        collection = self.db.get_collection('fs.files')
        if len(collection.index_information()) < 2:
            collection.create_index('set_id')


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
    def _get_doc(self, crf):
        return {
            'orig_indexes': crf.orig_indexes,
            'model_filename': crf.model_filename
        }

    def _doc_to_model(self, doc):
        data = eval(doc.read())
        crf = self._get_model_instance()
        crf.set_params(data['orig_indexes'], data['model_filename'])
        return crf

    def _get_model_instance(self):
        if self.method in [Method.LONG_CRF, Method.MULTI_STATE_LONG_CRF]:
            return CRF()
        elif self.method in [Method.BIN_CRF, Method.MULTI_STATE_BIN_CRF]:
            return BinCRF()
        elif self.method in [Method.TD_CRF, Method.MULTI_STATE_TD_CRF]:
            return TDCRF()
        else:
            raise ValueError(f"Invalid method {self.method.value} for CRF manager")
