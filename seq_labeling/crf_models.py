import numpy as np

from db.managers import CRFManager
from diffusion.enum import Method
from seq_labeling.models import NodeSeqLabelModel
from seq_labeling.pgm import CRF
from settings import logger


class CRFModel(NodeSeqLabelModel):
    method = Method.CRF

    @classmethod
    def _train_model(cls, evidence, iterations, key, graph, **kwargs):
        crf = CRF()
        crf.fit(evidence, iterations, **kwargs)
        return crf

    @classmethod
    def _get_seq_label_manager(cls, project):
        return CRFManager(project, cls.method)

    def _get_predicted_node_id(self, obs, model, tree, obs_node_ids):
        """ Set the parent with the maximum state feature coefficient which is also activated at the current step as the
        predicted parent of this child. """

        conv_indexes = [model.orig_indexes_map.get(ind) for ind in np.where(obs[0, :])[0]]
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
