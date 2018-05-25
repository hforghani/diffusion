import logging
import random
import time
from datetime import timedelta
import numpy as np
from cascade.models import CascadeNode, CascadeTree, ParamTypes
from crud.models import UserAccount
from memm.memm import MEMM
from utils.time_utils import str_to_datetime, DT_FORMAT


logger = logging.getLogger('diffusion.memm.models')


class MEMMModel():
    def __init__(self, project):
        self.project = project
        self.tree = None

    def fit(self, tree):
        """
        Set the tree of initial activated nodes.
        :param tree:    An instance of CascadeTree containing initial activated nodes
        :return:        self
        """
        if not isinstance(tree, CascadeTree):
            raise ValueError('tree must be CascadeTree')
        self.tree = tree.copy()
        return self

    def predict(self, user_ids=None, log=False):
        """
        Predict activation cascade in the future starting from initial nodes in self.tree.
        Set the final tree again in self.tree.
        :param user_ids: List of possible users for activation. All of users if users_ids is None.
        :param log:      Log in console if True else does not log.
        :return:         Returns self.tree
        """
        if not self.tree:
            raise ValueError('fit a data before prediction')

        # Initialize values.
        t0 = time.time()
        now = self.tree.max_datetime()  # Find the datetime of now.
        cur_step = sorted(self.tree.nodes(), key=lambda n: n.datetime)  # Set tree nodes as initial step.
        activated = self.tree.nodes()
        self.weight_sum = {}
        if user_ids is None:
            user_ids = UserAccount.objects.values_list('id', flat=True).order_by('id')
        user_map = {user_ids[i]: i for i in range(len(user_ids))}
        if log:
            logger.info('time1 = %.2f' % (time.time() - t0))

        memms = []
        for node in activated:
            m = MEMM()
            m.fit(observations, states)
            memms.append(m)

        return self.tree
