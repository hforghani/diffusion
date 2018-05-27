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

    def fit(self, user_ids):
        """
        Set the tree of initial activated nodes.
        :param user_ids:    Users which we want to set a MEMM for each of them.
        :return:        self
        """
        memms = []
        for user_id in user_ids:
            m = MEMM()
            m.fit(observations, obs_dim)
            memms.append(m)

        return self

    def predict(self, initial_tree, user_ids=None, log=False):
        """
        Predict activation cascade in the future starting from initial nodes in initial_tree.
        :param user_ids: List of possible users for activation. All of users is considered if users_ids is None.
        :param log:      Log in console if True else does not log.
        :return:         Predicted tree
        """
        if not isinstance(initial_tree, CascadeTree):
            raise ValueError('tree must be CascadeTree')
        tree = initial_tree.copy()

        # Initialize values.
        t0 = time.time()
        #now = tree.max_datetime()  # Find the datetime of now.
        cur_step = sorted(tree.nodes(), key=lambda n: n.datetime)  # Set tree nodes as initial step.
        activated = tree.nodes()
        self.weight_sum = {}
        if user_ids is None:
            user_ids = UserAccount.objects.values_list('id', flat=True).order_by('id')
        user_map = {user_ids[i]: i for i in range(len(user_ids))}
        if log:
            logger.info('time1 = %.2f' % (time.time() - t0))

        return tree
