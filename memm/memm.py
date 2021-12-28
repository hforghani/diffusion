from functools import reduce
from typing import Tuple
import numpy as np
import scipy
from scipy.sparse import csr_matrix

from memm.exceptions import MemmException
from settings import logger
from utils.time_utils import Timer, TimeUnit, time_measure


def obs_to_str(obs: int, dim: int) -> str:
    return '{:>0{w}}'.format(bin(obs)[2:], w=dim)[::-1]


def obs_to_array(obs: int, dim: int) -> np.array:
    obs_arr = []
    for _ in range(dim):
        obs_arr.append(obs % 2)
        obs >>= 1
    return np.array(obs_arr, dtype=bool)


def array_to_obs(arr: np.array) -> int:
    obs = 0
    for index in np.nonzero(arr)[0]:
        obs |= 1 << int(index)
    return obs


def array_to_str(arr: np.array) -> str:
    return ''.join(str(int(d)) for d in arr)


class MEMM:
    def __init__(self):
        self.orig_indexes = []
        self.Lambda = None

    @time_measure(level='debug')
    def fit(self, evidence):
        """
        Learn MEMM lambdas and transition probabilities for each previous state.
        :param evidence:   an instance of MemmEvidence
        :return:            self
        """
        dim, sequences = evidence['dimension'], evidence['sequences']
        new_sequences, self.orig_indexes = self.decrease_dim(sequences, dim)
        new_dim = len(self.orig_indexes)

        if new_dim == 0:
            raise MemmException('Cannot train MEMM with all observations given zero')

        all_obs = reduce(lambda s1, s2: s1 | s2, [{pair[0] for pair in seq} for seq in new_sequences])
        all_obs = list(all_obs)
        all_obs_arr = np.array([obs_to_array(obs, new_dim) for obs in all_obs])
        logger.debugv('all_obs_arr = %s', all_obs_arr)
        map_obs_index = {v: k for k, v in dict(enumerate(all_obs)).items()}

        # Create matrices of observations and states of which previous state is zero.
        obs_mat, state_mat, obs_indexes = self.__create_matrices(new_sequences, all_obs_arr, map_obs_index)

        # If there is no state=1, set probabilities manually.
        if not np.any(state_mat):
            return self

        # Calculate features for observation-state pairs. Shape of f1 is obs_num * (obs_dim+1)
        features, C = self.__calc_features(obs_mat, state_mat)

        # Calculate the training data average for each feature.
        F = np.mean(features, axis=0).T
        logger.debugv('F =\n%s', F)

        # Initialize Lambda as 1 then learn from training data.
        # Lambda is different per s' (previous state), But here just we use s' = 0.
        self.Lambda = np.ones(new_dim + 1)

        # GIS, run until convergence
        epsilon = 10 ** -4
        max_iteration = 1000
        iter_count = 0
        while True:
            iter_count += 1
            logger.debugv("iteration = %d ...", iter_count)
            Lambda0 = np.copy(self.Lambda)
            TPM = self.__build_tpm(self.Lambda, all_obs_arr)
            logger.debugv('TPM =\n%s', TPM)
            E = self.__build_expectation(obs_mat, TPM, obs_indexes)
            logger.debugv('E =\n%s', E)
            self.Lambda = self.__build_next_lambda(self.Lambda, C, F, E)
            logger.debugv('lambda = %s', self.Lambda)

            diff = np.linalg.norm(Lambda0 - self.Lambda) / np.sqrt(new_dim + 1)
            if diff < epsilon or iter_count >= max_iteration:
                logger.debug('GIS iterations = %d, diff = %s', iter_count, diff)
                break

        return self

    def get_prob(self, obs: np.ndarray, timers: list = None) -> float:
        """
        Get the probability of state=1 conditioned on the given observation and the previous state = 0 (inactivated).
        :param obs:     current observation
        """
        # if timers is None:
        #     timers = [Timer(f'get_prob part {i}', level='debug', unit=TimeUnit.SECONDS) for i in range(2)]

        # with timers[0]:
        dim = len(self.orig_indexes)
        obs_mat = np.tile(obs, (2, 1))
        f, _ = self.__calc_features(obs_mat, np.array([[False], [True]]))
        values = f.dot(self.Lambda.reshape(dim + 1, 1))
        prob = float(scipy.special.expit(values[1] - values[0]))

        # logger.debugv('obs = %s', obs)
        # logger.debugv('new_dim = %s', dim)
        # logger.debugv('obs_mat = %s', obs_mat)
        # logger.debugv('f = %s', f)
        # logger.debugv('values = %s', values)
        # logger.debugv('prob = %s', prob)
        return prob

    def predict(self, obs: int, threshold: float) -> Tuple[int, float]:
        """
        Predict the state conditioned on the given observation if the previous state is 0 (inactivated).
        :param threshold: probability threshold
        :param obs:     current observation
        :return:        predicted next state
        """
        prob = self.get_prob(obs)
        next_state = int(prob >= threshold)
        return next_state, prob

    def __create_matrices(self, sequences, all_obs_arr, map_obs_index):
        obs_indexes = []
        states = []
        for seq in sequences:
            if not seq:
                continue
            obs_indexes.extend(map_obs_index[obs] for obs, state in seq[1:])
            states.extend(state for obs, state in seq[1:])
        obs_mat = all_obs_arr[obs_indexes, :]
        states_array = np.array(states, dtype=bool)
        return obs_mat, states_array, obs_indexes

    @staticmethod
    def __calc_features(obs_mat, state_mat):
        obs_num, obs_dim = obs_mat.shape
        # features = np.logical_and(obs_mat, np.tile(np.reshape(state_mat, (obs_num, 1)), obs_dim))
        features = np.logical_not(
            np.logical_xor(obs_mat, np.tile(np.reshape(state_mat, (obs_num, 1)), obs_dim)))
        # C = np.max(np.sum(obs_mat, axis=1)) + 1  # C is chosen so that is greater than sum of any row.
        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        return features, C

    def __build_tpm(self, Lambda, all_obs):
        """
        Create normalized transition probability matrix (TPM) from previous state of 0 (inactivated) given current observation
        :param Lambda:      np array of Lambda weights
        :param all_obs:     np array of all unique observations: obs_num * obs_dim
        :return:            np array of shape (obs_num, 2)
        """
        obs_num = all_obs.shape[0]
        # f0 is features of observations with state = 0.
        state_mat = np.zeros((obs_num, 1), dtype=bool)
        f0, _ = self.__calc_features(all_obs, state_mat)

        # f1 is features of observations with state = 1.
        state_mat = np.ones((obs_num, 1), dtype=bool)
        f1, _ = self.__calc_features(all_obs, state_mat)

        P0 = np.squeeze(f0.dot(Lambda))
        P1 = np.squeeze(f1.dot(Lambda))
        normalized0 = np.reshape(scipy.special.expit(P0 - P1), (obs_num, 1))
        normalized1 = 1 - normalized0
        TPM = np.concatenate((normalized0, normalized1), axis=1)

        return TPM

    def __build_expectation(self, obs_mat, TPM, obs_indexes):
        obs_num, obs_dim = obs_mat.shape

        # f0 is features of observations with state = 0.
        state_mat = np.zeros((obs_num, 1), dtype=bool)
        f0, _ = self.__calc_features(obs_mat, state_mat)

        # f1 is features of observations with state = 1.
        state_mat = np.ones((obs_num, 1), dtype=bool)
        f1, _ = self.__calc_features(obs_mat, state_mat)

        obs_probs = TPM[obs_indexes, :]
        E = obs_probs[:, 0].T.dot(f0) + obs_probs[:, 1].T.dot(f1)
        E /= obs_num
        return E

    @staticmethod
    def __build_next_lambda(Lambda, C, F, E):
        """
        Use Generalized iterative scaling (GIS) to learn Lambda parameter
        """
        Lambda += (np.log(F) - np.log(E)) / C
        Lambda[F == 0] = 0
        return Lambda

    def decrease_dim(self, sequences, dim):
        """
        Decrease dimensions of observations in sequences. Remove the dimensions related to the parents
        which has no activation (e.t. has no digit 1) in any observation.
        :param sequences:   list of sequences of (obs, state)
        :param dim:         number of observation dimensions
        :return:            new sequences, map of new indexes to the old ones; with one difference that
                            the observation is numpy array in tuple (obs, state)
        """
        # Find the dimensions with any non-zero value.
        has_nonzero = 0
        for seq in sequences:
            for obs, state in seq[1:]:
                has_nonzero |= obs

        # Extract indexes of non-zero values of observations. These are the dimension we want to preserve.
        orig_indexes = []
        for i in range(dim):
            if has_nonzero % 2:
                orig_indexes.append(i)
            has_nonzero >>= 1
            if has_nonzero == 0:
                break

        # Count the used (nonzero) dimensions
        new_dim = len(orig_indexes)

        if new_dim == dim:
            return sequences, list(range(dim))

        # Decrease the dimensions and create the new sequences.
        new_sequences = [[(self.decrease_dim_by_indexes(obs, orig_indexes), state) for obs, state in seq] for seq in
                         sequences]

        return new_sequences, orig_indexes

    @staticmethod
    def decrease_dim_by_indexes(obs: int, orig_indexes: list) -> int:
        new_obs = 0
        for ind in orig_indexes[::-1]:
            new_obs <<= 1
            new_obs += (obs >> ind) % 2
        return new_obs
