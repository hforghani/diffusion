import abc
from abc import ABC
from functools import reduce
import numpy as np

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
    if arr.dtype == bool:
        return ''.join(str(int(d)) for d in arr)
    else:
        return str(arr.tolist())


class MEMM(abc.ABC):
    def __init__(self):
        self.orig_indexes = []
        self.orig_indexes_map = {}
        self.Lambda = None

    @time_measure(level='debug')
    def fit(self, evidence: dict, states: list, iterations: int):
        """
        Learn MEMM lambdas and transition probabilities for previous state of 0.
        :param evidence:   dictionary of sequences in the format {'dimension': int, 'sequences': list}
        :param states:      list of all possible states
        :param iterations: maximum iterations
        :return:            self
        """
        dim, sequences = evidence['dimension'], evidence['sequences']
        new_sequences, self.orig_indexes = self.decrease_dim(sequences, dim)
        new_dim = len(self.orig_indexes)
        self.orig_indexes_map = {self.orig_indexes[i]: i for i in range(len(self.orig_indexes))}

        if new_dim == 0:
            raise MemmException('Cannot train MEMM with all observations given zero')

        all_obs_arr, map_obs_index = self.get_all_obs_mat(new_sequences)
        logger.debugv('all_obs_arr =\n%s', all_obs_arr)

        # Create matrices of observations and states of which previous state is zero.
        obs_mat, state_mat, obs_indexes = self._create_matrices(new_sequences, map_obs_index)
        logger.debugv('all possible states = %s', states)
        logger.debugv('obs_mat, state_mat =\n%s',
                      np.concatenate((obs_mat, state_mat.reshape(state_mat.shape[0], 1)), axis=1))

        # Calculate features for observation-state pairs. Shape of features is obs_num * (obs_dim+1)
        features, C = self._calc_features(obs_mat, state_mat, states)
        logger.debugv('features =\n%s', features)
        feat_dim = features.shape[1]

        # Calculate the training data average for each feature.
        F = np.mean(features, axis=0).T
        logger.debugv('F =\n%s', F)

        # Initialize Lambda as 1 then learn from training data.
        # Lambda is different per s' (previous state), But here just we use s' = 0.
        self.Lambda = np.ones(feat_dim)

        # GIS, run until convergence
        epsilon = 10 ** -10
        iter_count = 0
        while True:
            iter_count += 1
            logger.debugv("iteration = %d ...", iter_count)
            Lambda0 = np.copy(self.Lambda)
            TPM = self.__build_tpm(self.Lambda, all_obs_arr, states)
            logger.debugv('TPM =\n%s', TPM)
            E = self.__build_expectation(obs_mat, TPM, obs_indexes, states)
            logger.debugv('E =\n%s', E)
            self.Lambda = self.__build_next_lambda(self.Lambda, C, F, E)
            logger.debugv('lambda = %s', self.Lambda)

            diff = np.linalg.norm(Lambda0 - self.Lambda) / np.sqrt(feat_dim)
            if diff < epsilon or iter_count >= iterations:
                logger.debug('GIS iterations = %d, diff = %s', iter_count, diff)
                break

        return self

    def set_params(self, Lambda, orig_indexes):
        self.Lambda = Lambda
        self.orig_indexes = orig_indexes
        self.orig_indexes_map = {self.orig_indexes[i]: i for i in range(len(self.orig_indexes))}

    def get_prob(self, obs: np.ndarray, state: int, all_states: list, timers: list = None) -> float:
        """
        Get the probability of state=1 conditioned on the given observation and the previous state = 0 (inactivated).
        :param obs:     current observation
        :param state:   the state
        :param all_states:  list of all possible states
        """
        # if timers is None:
        #     timers = [Timer(f'get_prob part {i}', level='debug', unit=TimeUnit.SECONDS) for i in range(2)]

        # with timers[0]:
        obs_mat = np.tile(obs, (len(all_states), 1))
        f, _ = self._calc_features(obs_mat, np.array(all_states).reshape((len(all_states), 1)), all_states)
        dots = f.dot(self.Lambda.reshape(self.Lambda.shape[0], 1))
        s_value = dots[all_states.index(state)]
        prob = float(1 / np.sum(np.array([np.exp(dots[i] - s_value) for i in range(len(all_states))])))
        if (obs == 0).all():
            logger.debug('obs = 0 -> prob = %f', prob)
        return prob

    def get_probs(self, obs: np.ndarray, all_states: list) -> list:
        """
        Predict the state of the observation given conditioned on that the previous state is 0 (inactive).
        :param obs:         current observation
        :param all_states:  list of all possible states
        :return:            list of probabilities related to the states given
        """
        obs_mat = np.tile(obs, (len(all_states), 1))
        f, _ = self._calc_features(obs_mat, np.array(all_states).reshape((len(all_states), 1)), all_states)
        dots = f.dot(self.Lambda.reshape(self.Lambda.shape[0], 1))
        probs = np.zeros(len(all_states))
        for i in range(len(all_states)):
            exp_sum = np.sum(np.array([np.exp(dots[j] - dots[i]) for j in range(len(all_states)) if j != i]))
            probs[i] = 1 / (1 + exp_sum)
        # logger.debug('obs = %s', obs)
        # logger.debug('states = %s', states)
        # logger.debug('all_states = %s', all_states)
        # logger.debug('features =\n%s', f)
        # logger.debug('self.Lambda = %s', self.Lambda)
        # logger.debug('dots =\n%s', dots)
        # logger.debug('probs =%s', probs)
        return probs.tolist()

    def get_all_obs_mat(self, sequences):
        all_obs = list(reduce(lambda li1, li2: li1 + li2, [[pair[0] for pair in seq] for seq in sequences]))
        all_obs_arr = np.unique(np.array(all_obs), axis=0)
        map_obs_index = {tuple(all_obs_arr[i, :]): i for i in range(all_obs_arr.shape[0])}
        return all_obs_arr, map_obs_index

    def _create_matrices(self, sequences, map_obs_index):
        obs_indexes = []
        states = []
        obs_mat = []
        for seq in sequences:
            if not seq:
                continue
            obs_mat.extend(obs for obs, state in seq[1:])
            obs_indexes.extend(map_obs_index[tuple(obs)] for obs, state in seq[1:])
            states.extend(state for obs, state in seq[1:])
        obs_mat = np.array(obs_mat)
        states = np.array(states)
        return obs_mat, states, obs_indexes

    @abc.abstractmethod
    def _calc_features(self, obs_mat: np.ndarray, state_mat: np.ndarray, states: list = None):
        pass

    def __build_tpm(self, Lambda: np.ndarray, all_obs: np.ndarray, states: list):
        """
        Create normalized transition probability matrix (TPM) from previous state of 0 (inactivated) given current observation
        :param Lambda:      np array of Lambda weights
        :param all_obs:     np array of size obs_num * (d+1) containing all unique observations
        :param states:      list of all possible states
        :return:            np array of shape (obs_num, S) where S is number of states
        """
        obs_num = all_obs.shape[0]

        # features is a list of N * (d+1) matrices. features[i] contains the features of f(o, states[i] | s' = 0)
        # where o is all_obs.
        features = [self._calc_features(all_obs, np.ones((obs_num, 1)) * s, states)[0] for s in states]
        # logger.debugv('features =\n%s', pprint.pformat(features))

        # exp_dots is a list of values exp( f(o,s) . lambda ) for different s values
        dots = [np.squeeze(features[i].dot(Lambda)) for i in range(len(states))]
        # logger.debugv('dots =\n%s', pprint.pformat(dots))

        TPM = np.zeros((obs_num, len(states)))
        for i in range(len(states)):
            exp_sum = np.sum(np.array([np.exp(dots[j] - dots[i]) for j in range(len(states)) if j != i]), axis=0).T
            TPM[:, i] = 1 / (1 + exp_sum)

        return TPM

    def __build_expectation(self, obs_mat: np.ndarray, TPM: np.ndarray, obs_indexes: list, states: list):
        obs_num, obs_dim = obs_mat.shape

        # features is a list of N * (d+1) matrices. features[i] contains the features of f(o, states[i] | s' = 0)
        # where o is obs_mat.
        features = [self._calc_features(obs_mat, np.ones((obs_num, 1)) * s, states)[0] for s in states]
        # logger.debugv('features =\n%s', pprint.pformat(features))

        obs_probs = TPM[obs_indexes, :]
        # logger.debugv('obs_probs =\n%s', obs_probs)

        E = np.sum(np.array([features[i].T.dot(obs_probs[:, i]) for i in range(len(states))]), axis=0)
        E /= obs_num

        return E

    @staticmethod
    def __build_next_lambda(Lambda, C, F, E):
        """
        Use Generalized iterative scaling (GIS) to learn Lambda parameter
        """
        fcopy = F.copy()
        ecopy = E.copy()
        fcopy[F == 0] = 10 ** -10
        ecopy[E == 0] = 10 ** -10
        Lambda += (np.log(fcopy) - np.log(ecopy)) / C
        # Lambda[F == 0] = 0
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
        all_obs_mat, map_obs_index = self.get_all_obs_mat(sequences)
        union = all_obs_mat.any(axis=0)
        orig_indexes = list(np.nonzero(union)[0])
        new_dim = len(orig_indexes)

        if new_dim == dim:
            return sequences, list(range(dim))

        # Decrease the dimensions and create the new sequences.
        new_sequences = [[(obs[orig_indexes], state) for obs, state in seq] for seq in sequences]

        return new_sequences, orig_indexes


class BinMEMM(MEMM):
    def _calc_features(self, obs_mat, state_mat, states=None):
        if obs_mat.shape[0] != state_mat.shape[0]:
            raise ValueError('number of observations and sates must be equal')
        obs_num, obs_dim = obs_mat.shape
        # features = np.logical_and(obs_mat, np.tile(np.reshape(state_mat, (obs_num, 1)), obs_dim))
        features = np.logical_not(
            np.logical_xor(obs_mat, np.tile(np.reshape(state_mat.astype(bool), (obs_num, 1)), obs_dim)))
        # C = np.max(np.sum(obs_mat, axis=1)) + 1  # C is chosen so that is greater than sum of any row.
        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        return features, C


class TDMEMM(MEMM):
    def _calc_features(self, obs_mat, state_mat, states=None):
        if obs_mat.shape[0] != state_mat.shape[0]:
            raise ValueError('number of observations and sates must be equal')
        obs_num, obs_dim = obs_mat.shape
        features = obs_mat.copy()
        zero_state_indexes = np.where(state_mat == False)
        features[zero_state_indexes, :] = 1 - features[zero_state_indexes, :]
        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        return features, C

    def multi_prob(self, observations):
        obs_num = observations.shape[0]
        f0, _ = self._calc_features(observations, np.array([[False]] * obs_num), [False, True])
        f1, _ = self._calc_features(observations, np.array([[True]] * obs_num), [False, True])
        dots0 = f0.dot(self.Lambda.reshape(self.Lambda.shape[0], 1))
        dots1 = f1.dot(self.Lambda.reshape(self.Lambda.shape[0], 1))
        prob = 1 / (np.exp(dots0 - dots1) + 1)
        prob[(observations == 0).all(axis=1)] = 0
        # logger.debugv('observations =\n%s', observations)
        # logger.debugv('f0 =\n%s', f0)
        # logger.debugv('f1 =\n%s', f1)
        # logger.debugv('self.Lambda =\n%s', self.Lambda)
        # logger.debugv('prob =\n%s', prob)
        return prob


class ParentTDMEMM(MEMM):

    def _calc_features(self, obs_mat, state_mat, states=None):
        if obs_mat.shape[0] != state_mat.shape[0]:
            raise ValueError('number of observations and sates must be equal')
        if states is None:
            raise ValueError('states must be given for ParentTDMEMM')
        # logger.debugv('obs_mat, state_mat =\n%s',
        #               np.concatenate((obs_mat, state_mat.reshape(state_mat.shape[0], 1)), axis=1))
        # logger.debugv('states =\n%s', states)
        obs_num, obs_dim = obs_mat.shape
        features = np.zeros((obs_num, 2 * obs_dim))
        # logger.debugv('features of state 0 update :\n%s', features)

        state_to_index = {self.orig_indexes[i] + 1: i for i in range(obs_dim)}
        # logger.debugv('state_to_index = %s', state_to_index)
        for i in range(obs_num):
            s = int(state_mat[i])
            if s == 0:
                features[i, :obs_dim] = 1 - obs_mat[i, :]
            elif s in state_to_index:
                ind = state_to_index[s]
                features[i, obs_dim + ind] = obs_mat[i, ind]

        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        # logger.debugv('features =\n%s', features)
        return features, C


class LongParentTDMEMM(MEMM):

    def _calc_features(self, obs_mat, state_mat, states=None):
        if states is None:
            raise ValueError('states must be given for ParentTDMEMM')
        # logger.debugv('obs_mat, state_mat =\n%s',
        #               np.concatenate((obs_mat, state_mat.reshape(state_mat.shape[0], 1)), axis=1))
        # logger.debugv('states =\n%s', states)
        obs_num, obs_dim = obs_mat.shape
        features = np.zeros((obs_num, (obs_dim + 1) * obs_dim))
        # logger.debugv('features of state 0 update :\n%s', features)

        state_to_index = {self.orig_indexes[i] + 1: i for i in range(obs_dim)}
        # logger.debugv('state_to_index = %s', state_to_index)
        for i in range(obs_num):
            s = int(state_mat[i])
            if s == 0:
                features[i, :obs_dim] = 1 - obs_mat[i, :]
            elif s in state_to_index:
                ind = state_to_index[s]
                start = (ind + 1) * obs_dim
                features[i, start: start + obs_dim] = obs_mat[i, :]

        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        # logger.debugv('features =\n%s', features)
        return features, C
