import abc
import os

import numpy as np
import sklearn_crfsuite

from db.managers import CRFManager
from diffusion.enum import Method
from seq_labeling.models import NodeSeqLabelModel, ParentSensMultiStateModel, FullMultiStateModel
from seq_labeling.pgm import CRF, BinCRF, TDCRF
from settings import logger


class CRFModel(NodeSeqLabelModel):
    method = Method.LONG_CRF

    def __init__(self, algorithm='lbfgs', c1=0, c2=1, initial_depth=0, max_step=None, threshold=0.5,
                 keep_temp_files=True, **kwargs):
        super().__init__(initial_depth, max_step, threshold, **kwargs)
        self.algorithm = algorithm
        self.c1 = c1
        self.c2 = c2
        self.keep_temp_files = keep_temp_files
        self.__fetch_freq = {}  # Number of fetching of CRF (_get_model call) for each node id
        self.__node_ids_in_mem = set()  # The node ids of which CRF exists in the memory
        self.__min_fetch_freq = None
        self.__min_fetch_node_id = None
        self.__max_crf_in_mem = 1000  # Maximum number of CRF kept in memory

    @classmethod
    def train_model(cls, sequences, iterations, states, node_id, project, eco=False, **kwargs):
        # logger.debug('kwargs = %s', kwargs)
        dir_path = os.path.join(project.path, 'crf')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        model_filename = os.path.join(dir_path, f'{cls.method.value}-{str(node_id)}.crfsuite') if eco else None
        crf = cls.get_crf_instance(model_filename=model_filename, **kwargs)
        # logger.debug('training CRF ...')
        crf.fit(sequences, iterations, **kwargs)
        # logger.debug('CRF training done')

        crf.model_filename = crf.crf.modelfile.name
        cls.__clear_mem(crf)
        logger.debug('node id %s -> model_filename = %s', str(node_id), model_filename)

        return crf

    def _get_model(self, node_id):
        try:
            if node_id not in self._models:
                return None
            self.__fetch_freq.setdefault(node_id, 0)
            self.__fetch_freq[node_id] += 1
            # logger.debug('self.__fetch_freq[node_id] = %s', self.__fetch_freq[node_id])

            crf = self._models[node_id]

            if crf.crf is not None:
                # logger.debug('fetched CRF returned')
                return crf
            else:
                # logger.debug('fetching CRF ...')
                new_crf = self.get_crf_instance()
                new_crf.set_params(crf.orig_indexes, crf.model_filename)
                new_crf.crf = sklearn_crfsuite.CRF(model_filename=crf.model_filename, keep_tempfiles=True)

                # logger.debug('len(self.__node_ids_in_mem) = %s', len(self.__node_ids_in_mem))
                # logger.debug('self.__min_fetch_freq = %s', self.__min_fetch_freq)
                # logger.debug('self.__min_fetch_node_id = %s', self.__min_fetch_node_id)
                if self.__fetch_freq and len(self.__node_ids_in_mem) >= self.__max_crf_in_mem:
                    if self.__fetch_freq[node_id] > self.__min_fetch_freq:
                        self._models[node_id] = new_crf
                        self.__clear_mem(self._models[self.__min_fetch_node_id])
                        self.__node_ids_in_mem.remove(self.__min_fetch_node_id)
                        self.__node_ids_in_mem.add(node_id)
                        self.__min_fetch_freq, self.__min_fetch_node_id = self.__min_fetch_freq_in_mem()
                        # logger.debug('fetched CRF replaced')
                else:
                    self._models[node_id] = new_crf
                    self.__node_ids_in_mem.add(node_id)
                    # logger.debug('Max threshold not reached. CRF kept.')
                    if self.__min_fetch_freq is None or self.__fetch_freq[node_id] < self.__min_fetch_freq:
                        self.__min_fetch_freq = self.__fetch_freq[node_id]
                        self.__min_fetch_node_id = node_id
                        # logger.debug('min fetch freq replaced')

                return new_crf
        except:
            # logger.debug('self.__fetch_freq = %s', self.__fetch_freq)
            # logger.debug('self.__node_ids_in_mem = %s', self.__node_ids_in_mem)
            raise

    def __min_fetch_freq_in_mem(self):
        min_freq, min_node_id = None, None
        for node_id in self.__node_ids_in_mem:
            freq = self.__fetch_freq[node_id]
            if min_freq is None or freq < min_freq:
                min_freq = freq
                min_node_id = node_id
        return min_freq, min_node_id

    @classmethod
    def get_crf_instance(cls, **kwargs):
        return CRF(kwargs.get('model_filename'))

    @classmethod
    def _get_seq_label_manager(cls, project):
        return CRFManager(project, cls.method)

    def _predict_parent_id(self, obs_seq, model, tree, obs_node_ids):
        """ Set the parent with the maximum state feature coefficient which is also activated at the current step as the
        predicted parent of this child. """

        conv_indexes = [model.orig_indexes_map.get(ind) for ind in np.where(obs_seq[0, :])[0]]
        conv_indexes = list(filter(lambda x: x is not None, conv_indexes))

        if conv_indexes:
            weights = {}
            for (attr, state), weight in model.crf.state_features_.items():
                if state == '1':
                    dif, parent_index = attr.split(':')
                    if dif == '0' and parent_index in conv_indexes:
                        weights[int(parent_index)] = weight

            if weights:
                pred_parent_index = max(weights, key=lambda ind: weights[ind])
            else:
                pred_parent_index = conv_indexes[0]
            node_id = obs_node_ids[model.orig_indexes[pred_parent_index]]
            if tree.get_node(node_id):
                return node_id
            else:
                logger.warning('parent node %s does not exist', node_id)
        else:
            logger.debug('the newly active nodes are not available in the training data')
        return

    @classmethod
    def __clear_mem(cls, crf: CRF):
        """
        Close the CRF file and set the crf field None.
        """
        if hasattr(crf.crf.modelfile, 'fd'):
            os.close(crf.crf.modelfile.fd)  # Close the file to avoid too much open files.
        crf.crf = None  # Set the crf None to clear the memory.

    def clean_temp_files(self):
        count = 0
        for crf in self._models.values():
            if crf.model_filename.startswith('/tmp'):
                try:
                    os.unlink(crf.model_filename)
                    count += 1
                except FileNotFoundError:
                    pass
        if count:
            logger.info('%d files cleaned up', count)

    def __del__(self):
        if not self.keep_temp_files:
            self.clean_temp_files()


class SmallFeatCRFModel(CRFModel, abc.ABC):
    """
    In the subclasses of this model, the length of features is equal to number of the parents.
    The keys of the feature dict are the parent indexes.
    """

    def _predict_parent_id(self, obs_seq, model, tree, obs_node_ids):
        """ Set the parent with the maximum state feature coefficient which is also activated at the current step as the
        predicted parent of this child. """

        conv_indexes = [model.orig_indexes_map.get(ind) for ind in np.where(obs_seq[0, :])[0]]
        conv_indexes = list(filter(lambda x: x is not None, conv_indexes))

        if conv_indexes:
            weights = {}
            # logger.debug('model.crf.state_features_.items() = \n%s', model.crf.state_features_.items())
            for (parent_index, state), weight in model.crf.state_features_.items():
                if state == '1' and parent_index in conv_indexes:
                    weights[int(parent_index)] = weight
            # logger.debug('weights = %s', weights)

            if weights:
                pred_parent_index = max(weights, key=lambda ind: weights[ind])
            else:
                pred_parent_index = conv_indexes[0]
            # logger.debug('pred_parent_index = %s', pred_parent_index)
            node_id = obs_node_ids[model.orig_indexes[pred_parent_index]]
            # logger.debug('node_id = %s', node_id)
            if tree.get_node(node_id):
                return node_id
            else:
                logger.warning('parent node %s does not exist', node_id)
        else:
            logger.debug('the newly active nodes are not available in the training data')
        return


class FullBinCRFModel(CRFModel):
    """
    CRF with short features and full observations.
    """
    method = Method.FULL_MULTI_STATE_BIN_CRF

    @classmethod
    def get_crf_instance(cls, **kwargs):
        return BinCRF(kwargs.get('model_filename'))


class BinCRFModel(SmallFeatCRFModel):
    """
    CRF with short features and small observations.
    """
    method = Method.BIN_CRF

    @classmethod
    def get_crf_instance(cls, **kwargs):
        return BinCRF(kwargs.get('model_filename'))


class TDCRFModel(SmallFeatCRFModel):
    method = Method.TD_CRF

    def __init__(self, algorithm='lbfgs', c1=0, c2=1, initial_depth=0, max_step=None, threshold=0.5, td_param=0.5,
                 **kwargs):
        super().__init__(algorithm, c1, c2, initial_depth, max_step, threshold)
        self.td_param = td_param

    @classmethod
    def get_crf_instance(cls, td_param, **kwargs):
        return TDCRF(td_param, kwargs.get('model_filename'))


class FullMSBinCRFModel(FullBinCRFModel, FullMultiStateModel):
    method = Method.FULL_MULTI_STATE_BIN_CRF


class ParentMSCRFModel(CRFModel, ParentSensMultiStateModel):
    method = Method.PAR_MULTI_STATE_LONG_CRF


class ParentMSBinCRFModel(BinCRFModel, ParentSensMultiStateModel):
    method = Method.PAR_MULTI_STATE_BIN_CRF


class ParentMSTDCRFModel(TDCRFModel, ParentSensMultiStateModel):
    method = Method.PAR_MULTI_STATE_TD_CRF
