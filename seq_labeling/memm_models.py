from db.managers import SeqLabelDBManager
from diffusion.enum import Method
from seq_labeling.models import SeqLabelDifModel, NodeSeqLabelModel, MultiStateModel
from seq_labeling.pgm import *
from settings import logger


class MEMMModel(SeqLabelDifModel, abc.ABC):
    max_iterations = 500

    @classmethod
    def _get_seq_label_manager(cls, project):
        return SeqLabelDBManager(project, cls.method)

    @classmethod
    def train_model(cls, evidence, iterations, states, node_id, project, eco=False, **kwargs):
        memm = cls.get_memm_instance(**kwargs)
        memm.fit(evidence, iterations, states)
        return memm

    @classmethod
    @abc.abstractmethod
    def get_memm_instance(cls, *args, **kwargs):
        pass


class NodeMEMMModel(NodeSeqLabelModel, MEMMModel, abc.ABC):
    pass


class LongMEMMModel(NodeMEMMModel):
    method = Method.LONG_MEMM

    # max_iterations = 1000

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold)

    @classmethod
    def get_memm_instance(cls, *args, **kwargs):
        return LongMEMM()

    def _get_predicted_node_id(self, obs_seq, model, tree, obs_node_ids):
        # Set the parent with the maximum value of Lambda which is also activated at the current step as the
        # predicted parent of this child.
        obs_dim = len(obs_node_ids)
        conv_indexes = [model.orig_indexes_map.get(ind) for ind in np.where(obs_seq[0, :obs_dim])[0]]
        conv_indexes = list(filter(lambda x: x is not None, conv_indexes))
        if conv_indexes:
            max_lambda_ind = np.argmax(model.Lambda[conv_indexes])
            node_id = obs_node_ids[model.orig_indexes[conv_indexes[max_lambda_ind]]]
            if tree.get_node(node_id):
                return node_id
            else:
                logger.warning('parent node %s does not exist', node_id)
        else:
            logger.debugv('the newly active nodes are not available in the training data')

        return None


class MultiStateLongMEMMModel(LongMEMMModel, MultiStateModel):
    method = Method.MULTI_STATE_LONG_MEMM

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold)


class BinMEMMModel(NodeMEMMModel):
    method = Method.BIN_MEMM

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold)

    @classmethod
    def get_memm_instance(cls, *args, **kwargs):
        return BinMEMM()


class MultiStateBinMEMMModel(BinMEMMModel, MultiStateModel):
    method = Method.MULTI_STATE_BIN_MEMM

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold)


class TDMEMMModel(NodeMEMMModel):
    """
    Time-Decay MEMM Model
    """
    method = Method.TD_MEMM

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, td_param=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold, **kwargs)
        self.td_param = td_param

    @classmethod
    def get_memm_instance(cls, td_param, *args, **kwargs):
        return TDMEMM(td_param)


class MultiStateTDMEMMModel(TDMEMMModel, MultiStateModel):
    method = Method.MULTI_STATE_TD_MEMM

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, td_param=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold, **kwargs)
        self.td_param = td_param


class ParentSensTDMEMMModel(MultiStateModel):
    """
    Parent-sensitive Time-Decay MEMM model
    """
    method = Method.PARENT_SENS_TD_MEMM

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, td_param=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold, **kwargs)
        self.td_param = td_param

    @classmethod
    def get_memm_instance(cls, td_param, *args, **kwargs):
        return ParentTDMEMM(td_param)


class LongParentSensTDMEMMModel(MultiStateModel):
    method = Method.LONG_PARENT_SENS_TD_MEMM

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, td_param=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold, **kwargs)
        self.td_param = td_param

    @classmethod
    def get_memm_instance(cls, td_param, *args, **kwargs):
        return LongParentTDMEMM(td_param)
