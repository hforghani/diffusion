import abc
import os

import numpy as np
import sklearn_crfsuite

from db.managers import CRFManager
from diffusion.enum import Method
from seq_labeling.models import NodeSeqLabelModel, MultiStateModel
from seq_labeling.pgm import CRF, BinCRF, TDCRF
from settings import logger


class CRFModel(NodeSeqLabelModel):
    method = Method.LONG_CRF

    def __init__(self, algorithm='lbfgs', c1=0, c2=1, initial_depth=0, max_step=None, threshold=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold, **kwargs)
        self.algorithm = algorithm
        self.c1 = c1
        self.c2 = c2

    @classmethod
    def train_model(cls, evidence, iterations, states, node_id, project, eco=False, **kwargs):
        logger.debug('kwargs = %s', kwargs)
        dir_path = os.path.join(project.path, 'crf')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        model_filename = os.path.join(dir_path, f'{cls.method.value}-{str(node_id)}.crfsuite') if eco else None
        crf = cls.get_crf_instance(model_filename=model_filename, **kwargs)
        crf.fit(evidence, iterations, **kwargs)

        crf.model_filename = crf.crf.modelfile.name
        os.close(crf.crf.modelfile.fd)  # Close the file to avoid too much open files.
        crf.crf = None  # Set the crf None to clear the memory.
        logger.debug('node id %s -> model_filename = %s', str(node_id), model_filename)

        return crf

    def _get_model(self, node_id):
        crf = self._models.get(node_id)
        if crf is None:
            return None
        elif crf.crf is None:
            new_crf = self.get_crf_instance()
            new_crf.set_params(crf.orig_indexes, crf.model_filename)
            new_crf.crf = sklearn_crfsuite.CRF(model_filename=crf.model_filename, keep_tempfiles=True)
            return new_crf
        else:
            return crf

    @classmethod
    def get_crf_instance(cls, **kwargs):
        return CRF(kwargs.get('model_filename'))

    @classmethod
    def _get_seq_label_manager(cls, project):
        return CRFManager(project, cls.method)

    def _get_predicted_node_id(self, obs_seq, model, tree, obs_node_ids):
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
            logger.debugv('the newly active nodes are not available in the training data')
        return

    def __del__(self):
        count = 0
        for crf in self._models.values():
            if crf.model_filename.startswith('/tmp') and os.path.exists(crf.model_filename):
                os.unlink(crf.model_filename)
                count += 1
        logger.debug('%d files cleaned up', count)


class SmallFeatCRFModel(CRFModel, abc.ABC):
    """
    In the subclasses of this model, the length of features is equal to number of the parents.
    The keys of the feature dict are the parent indexes.
    """

    def _get_predicted_node_id(self, obs_seq, model, tree, obs_node_ids):
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
            logger.debugv('the newly active nodes are not available in the training data')
        return


class BinCRFModel(SmallFeatCRFModel):
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


class MultiStateCRFModel(CRFModel, MultiStateModel):
    method = Method.MULTI_STATE_LONG_CRF


class MultiStateBinCRFModel(BinCRFModel, MultiStateModel):
    method = Method.MULTI_STATE_BIN_CRF


class MultiStateTDCRFModel(TDCRFModel, MultiStateModel):
    method = Method.MULTI_STATE_TD_CRF
