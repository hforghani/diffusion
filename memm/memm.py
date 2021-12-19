from typing import Tuple
import numpy as np
import scipy
from scipy.sparse import csr_matrix

from settings import logger
from utils.time_utils import time_measure


class MemmException(Exception):
    pass


def obs_to_str(obs: int, dim: int) -> str:
    return '{:>0{w}}'.format(bin(obs)[2:], w=dim)[::-1]


def obs_to_array(obs: int, dim: int) -> np.array:
    obs_arr = []
    for _ in range(dim):
        obs_arr.append(obs % 2)
        obs >>= 1
    return np.array(obs_arr, dtype=bool)


def obs_to_sparse(obs: int, dim: int) -> np.array:
    return csr_matrix(obs_to_array(obs, dim))


def array_to_obs(arr: np.array) -> int:
    obs = 0
    for index in np.nonzero(arr)[1]:
        obs |= 1 << int(index)
    return obs


def sparse_to_obs(arr: csr_matrix) -> int:
    return array_to_obs(arr.toarray())


def array_to_str(arr: np.array) -> str:
    return ''.join(str(int(d)) for d in arr)


class MEMM:
    def __init__(self):
        self.all_obs_arr = None
        self.orig_indexes = []
        self.map_obs_prob = {}

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

        all_obs = set()
        for seq in new_sequences:
            all_obs.update([pair[0] for pair in seq])
        all_obs = list(all_obs)
        self.all_obs_arr = csr_matrix(np.array([obs_to_array(obs, new_dim) for obs in all_obs]))
        map_obs_index = {v: k for k, v in dict(enumerate(all_obs)).items()}

        # Get pairs of (obs, state) which their previous state is 0.
        rel_pairs, rel_indexes = self.__get_related_pairs(new_sequences, map_obs_index)
        # logger.debugv('rel_pairs = %s', rel_pairs)

        # Create matrices of observations and states for related pairs.
        obs_mat, state_mat = self.__create_matrices(rel_pairs, rel_indexes)

        # If there is no state=1, set probabilities manually.
        if not np.any(state_mat):
            self.map_obs_prob = {obs: 0 for obs in all_obs}
            return self

        # Calculate features for observation-state pairs. Shape of f1 is obs_num * (obs_dim+1)
        features, C = self.__calc_features(obs_mat, state_mat)

        # Calculate the training data average for each feature.
        F = np.mean(features, axis=0).T

        # Initialize Lambda as 1 then learn from training data.
        # Lambda is different per s' (previous state), But here just we use s' = 0.
        Lambda = np.ones(new_dim + 1)

        # GIS, run until convergence
        epsilon = 10 ** -5
        max_iteration = 1000
        iter_count = 0
        while True:
            iter_count += 1
            logger.debugv("iteration = %d ...", iter_count)
            Lambda0 = np.copy(Lambda)
            TPM = self.__build_tpm(Lambda, self.all_obs_arr)
            logger.debugv('TPM =\n%s', TPM)
            E = self.__build_expectation(obs_mat, TPM, rel_indexes)
            logger.debugv('E =\n%s', E)
            Lambda = self.__build_next_lambda(Lambda, C, F, E)
            logger.debugv('lambda = %s', Lambda)

            diff = np.linalg.norm(Lambda0 - Lambda) / (new_dim + 1)
            if diff < epsilon or iter_count >= max_iteration:
                logger.debug('GIS iterations = %d, diff = %s', iter_count, diff)
                break

        for obs, index in map_obs_index.items():
            self.map_obs_prob[obs] = TPM[index][1]

        return self

    def get_prob(self, obs: int) -> float:
        """
        Get the probability of state=1 conditioned on the given observation and the previous state = 0 (inactivated).
        :param obs:     current observation
        """
        # logger.debug('running MEMM predict method ...')
        new_obs = self.decrease_dim_by_indexes(obs, self.orig_indexes)
        new_dim = len(self.orig_indexes)
        # new_obs_bin = obs_to_str(new_obs, new_dim)

        if new_obs in self.map_obs_prob:
            return self.map_obs_prob[new_obs]
        else:
            # prob = 0
            nearest_obs, sim = self.__nearest_obs(new_obs, new_dim)
            # logger.debug('obs %s not found. nearest: %s , sim = %f , prob = %f', obs_to_str(new_obs, new_dim),
            #              obs_to_str(nearest_obs, new_dim), sim, self.map_obs_prob[nearest_obs])
            # use probability * similarity when the observation not found.
            prob = self.map_obs_prob[nearest_obs] * sim

            return prob

    def predict(self, obs: int, threshold: float, timers=None) -> Tuple[int, float]:
        """
        Predict the state conditioned on the given observation if the previous state is 0 (inactivated).
        :param threshold: probability threshold
        :param obs:     current observation
        :return:        predicted next state
        """
        prob = self.get_prob(obs)
        next_state = int(prob >= threshold)
        return next_state, prob

    def __nearest_obs(self, obs: int, dim: int) -> Tuple[int, float]:
        """
        Get the nearest nonzero observation to the observation given.
        :param obs: observation as an int number
        :param dim: number of dimensions
        :return: a tuple of the nearest nonzero observation and similarity rate
        """
        observations = sorted(list(self.map_obs_prob.keys()))[1:]
        dist = np.array([bin(obs ^ o).count('1') for o in observations])
        index = np.argmin(dist)
        sim = (dim - np.min(dist)) / dim
        return observations[index], sim

    def __create_matrices(self, pairs, indexes):
        obs_mat = self.all_obs_arr[indexes, :]
        state_array = [pair[1] for pair in pairs]
        return obs_mat, np.array(state_array, dtype=bool)

    @staticmethod
    def __calc_features(obs_mat, state_mat):
        obs_num, obs_dim = obs_mat.get_shape()
        # features = np.logical_and(obs_mat, np.tile(np.reshape(state_mat, (obs_num, 1)), obs_dim))
        features = np.logical_not(
            np.logical_xor(obs_mat.toarray(), np.tile(np.reshape(state_mat, (obs_num, 1)), obs_dim)))
        # C = np.max(np.sum(obs_mat, axis=1)) + 1  # C is chosen so that is greater than sum of any row.
        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        return features, C

    def __get_related_pairs(self, sequences, map_obs_index):
        """
        Return related observation-state pairs of which the previous state is 0 (inactivated).
        :param sequences: list of sequences
        :return: list of tuples
        """
        rel_pairs = []
        rel_indexes = []

        for seq in sequences:
            if not seq:
                continue
            previous_state = seq[0][1]
            for pair in seq[1:]:
                if previous_state == 1:
                    break
                else:
                    rel_pairs.append(pair)
                    rel_indexes.append(map_obs_index[pair[0]])
                    previous_state = pair[1]

        return rel_pairs, rel_indexes

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

        TPM = np.zeros((obs_num, 2))
        TPM[:, 0] = np.squeeze(f0.dot(Lambda))
        TPM[:, 1] = np.squeeze(f1.dot(Lambda))
        normalized0 = np.reshape(scipy.special.expit(TPM[:, 0] - TPM[:, 1]), (obs_num, 1))
        normalized1 = np.reshape(scipy.special.expit(TPM[:, 1] - TPM[:, 0]), (obs_num, 1))
        TPM = np.concatenate((normalized0, normalized1), axis=1)

        return TPM

    def __build_expectation(self, obs_mat, TPM, tuple_indexes):
        obs_num, obs_dim = obs_mat.shape

        # f0 is features of observations with state = 0.
        state_mat = np.zeros((obs_num, 1), dtype=bool)
        f0, _ = self.__calc_features(obs_mat, state_mat)

        # f1 is features of observations with state = 1.
        state_mat = np.ones((obs_num, 1), dtype=bool)
        f1, _ = self.__calc_features(obs_mat, state_mat)

        tuples_TPM = TPM[tuple_indexes, :]
        E = tuples_TPM[:, 0].T.dot(f0) + tuples_TPM[:, 1].T.dot(f1)
        E /= obs_num
        return E

    @staticmethod
    def __build_next_lambda(Lambda, C, F, E):
        """
        Use Generalized iterative scaling (GIS) to learn Lambda parameter
        """
        for i in range(Lambda.size):
            # If the average for the feature is 0, it has no contribution to the probability
            if F[i] == 0:
                Lambda[i] = 0
            else:
                Lambda[i] += (np.log(F[i]) - np.log(E[i])) / C
        return Lambda

    @staticmethod
    def __check_lambda_convergence(Lambda0, Lambda1, epsilon):
        """
        Check if the lambdas are relatively the same.
        :param Lambda0: previous lambda
        :param Lambda1: current lambda
        :param epsilon: threshold of distance
        :return: True if the distance is lower than epsilon
        """
        return np.count_nonzero(np.absolute(Lambda0 - Lambda1) > epsilon) == 0

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
        has_nnz_copy = has_nonzero
        for i in range(dim):
            if has_nnz_copy % 2:
                orig_indexes.append(i)
            has_nnz_copy >>= 1
        orig_indexes.sort()

        # Count the used (nonzero) dimensions
        new_dim = len(orig_indexes)

        if new_dim == dim:
            return sequences, {i: i for i in range(dim)}

        # Decrease the dimensions and create the new sequences.
        new_sequences = []
        for seq in sequences:
            new_seq = []
            for obs, state in seq:
                new_obs = self.decrease_dim_by_indexes(obs, orig_indexes)
                new_seq.append((new_obs, state))
            new_sequences.append(new_seq)

        return new_sequences, orig_indexes

    @staticmethod
    def decrease_dim_by_indexes(obs, orig_indexes):
        new_obs = 0
        for ind in orig_indexes[::-1]:
            new_obs <<= 1
            new_obs += (obs >> ind) % 2
        return new_obs
