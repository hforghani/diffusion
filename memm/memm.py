import abc
import pprint
import typing
from abc import ABC
from functools import reduce
import numpy as np

from memm.exceptions import MemmException
from settings import logger
from utils.time_utils import Timer, TimeUnit, time_measure


def obs_to_str(arr: np.array) -> str:
    rows = []
    for row in arr:
        if row.dtype == bool:
            rows.append(''.join(str(int(d)) for d in row))
        else:
            rows.append(str(row.tolist()))
    return '\n'.join(rows)


def arr_to_str(arr: np.array) -> str:
    if arr.dtype == bool:
        return ''.join(str(int(d)) for d in arr)
    else:
        return str(arr.tolist())


def two_d_arr_to_str(arr: np.array) -> str:
    return '\n'.join(arr_to_str(arr[i, :]) for i in range(arr.shape[0]))


class MEMM(abc.ABC):
    def __init__(self):
        self.orig_indexes = []
        self.orig_indexes_map = {}
        self.Lambda = None
        self.feat_dim = None

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
        new_sequences, self.orig_indexes = self.decrease_dim(sequences)
        new_dim = len(self.orig_indexes)
        self.orig_indexes_map = {self.orig_indexes[i]: i for i in range(len(self.orig_indexes))}

        if new_dim == 0:
            raise MemmException('Cannot train MEMM with all observations given zero')

        # Get all observations and states of which previous state is zero.
        all_obs, state_mat = self.get_all_obs(new_sequences)
        logger.debugv('all_obs =\n%s', all_obs)
        logger.debugv('state_mat =\n%s', state_mat)
        logger.debugv('all possible states = %s', states)

        # Calculate features for observation-state pairs. Shape of features is obs_num * (obs_dim+1)
        features, C = self._calc_features(all_obs, state_mat)
        self.feat_dim = features.shape[1]
        logger.debugv('features =\n%s', features)

        obs_num = len(all_obs)
        features_for_all_states = [self._calc_features(all_obs, np.ones(obs_num, dtype=type(states[0])) * s)[0] for s in
                                   states]
        # logger.debugv('features_for_all_states =\n%s', pprint.pformat(features_for_all_states))

        # Calculate the training data average for each feature.
        F = np.mean(features, axis=0).T
        logger.debugv('F =\n%s', F)

        # Initialize Lambda as 1 then learn from training data.
        # Lambda is different per s' (previous state), But here just we use s' = 0.
        self.Lambda = np.ones(self.feat_dim)

        # GIS, run until convergence
        epsilon = 10 ** -10
        iter_count = 0
        while True:
            iter_count += 1
            logger.debugv("iteration = %d ...", iter_count)
            Lambda0 = np.copy(self.Lambda)
            TPM = self.__build_tpm(features_for_all_states, self.Lambda)
            logger.debugv('TPM =\n%s', TPM)
            E = self.__build_expectation(features_for_all_states, TPM)
            logger.debugv('E =\n%s', E)
            self.Lambda = self.__build_next_lambda(self.Lambda, C, F, E)
            logger.debugv('lambda = %s', self.Lambda)

            diff = np.linalg.norm(Lambda0 - self.Lambda) / np.sqrt(self.feat_dim)
            if diff < epsilon or iter_count >= iterations:
                logger.debug('GIS iterations = %d, diff = %s', iter_count, diff)
                break

        return self

    def set_params(self, Lambda, orig_indexes):
        self.Lambda = Lambda
        self.orig_indexes = orig_indexes
        self.orig_indexes_map = {self.orig_indexes[i]: i for i in range(len(self.orig_indexes))}
        self.feat_dim = Lambda.size

    def get_prob(self, obs: np.ndarray, state: int, all_states: list, timers: list = None) -> float:
        """
        Get the probability of state=1 conditioned on the given observation and the previous state = 0 (inactivated).
        :param obs:     current observation
        :param state:   the state
        :param all_states:  list of all possible states
        """
        # if timers is None:
        #     timers = [Timer(f'get_prob part {i}', level='debug', unit=TimeUnit.SECONDS) for i in range(2)]

        obs = obs[:, self.orig_indexes]
        if np.count_nonzero(obs) == 0:
            return 0
        f, _ = self._calc_features(obs, np.array(all_states))
        dots = f.dot(self.Lambda.reshape(self.Lambda.shape[0], 1))
        s_value = dots[all_states.index(state)]
        prob = float(1 / np.sum(np.array([np.exp(dots[i] - s_value) for i in range(len(all_states))])))
        return prob

    def get_probs(self, obs: np.ndarray, all_states: list) -> list:
        """
        Get the list of probabilities of transition to all states given the observation conditioned on that the previous
        state is inactive.
        :param obs:         current observation
        :param all_states:  list of all possible states
        :return:            list of probabilities related to the states given
        """
        obs = obs[:, self.orig_indexes]
        observations = [obs] * len(all_states)
        f, _ = self._calc_features(observations, np.array(all_states))
        dots = f.dot(self.Lambda.reshape(self.Lambda.shape[0], 1))
        probs = np.zeros(len(all_states))
        for i in range(len(all_states)):
            exp_sum = np.sum(np.array([np.exp(dots[j] - dots[i]) for j in range(len(all_states)) if j != i]))
            probs[i] = 1 / (1 + exp_sum)
        return probs.tolist()

    def get_multi_obs_probs(self, observations: list, all_states: list) -> np.ndarray:
        probs = np.zeros((len(observations), len(all_states)))
        for i in range(len(observations)):
            probs[i, :] = np.array(self.get_probs(observations[i], all_states))
        # logger.debug('observations = \n%s', observations)
        # logger.debug('probs = \n%s', probs)
        return probs

    def get_all_obs(self, sequences):
        observations = []
        states = []
        for seq in sequences:
            for obs, state in seq:
                observations.append(obs)
                states.append(state)
        states = np.array(states)
        return observations, states

    def _calc_features(self, obs: typing.Union[list, np.ndarray], states: np.ndarray) -> np.ndarray:
        if isinstance(obs, list):
            return self._calc_multi_obs_features(obs, states)
        else:
            return self._calc_obs_features(obs, states)

    def _calc_obs_features(self, obs: np.ndarray, states: np.ndarray) -> np.ndarray:
        return self._calc_multi_obs_features([obs] * len(states), states)

    @abc.abstractmethod
    def _calc_multi_obs_features(self, observations: list, states: np.ndarray) -> np.ndarray:
        pass

    def __build_tpm(self, features: list, Lambda: np.ndarray):
        """
        Create normalized transition probability matrix (TPM) from previous state of 0 (inactivated) given current observation
        :param Lambda:      np array of Lambda weights
        :param features:     list of lists. Each row of features[i] contains the features of f(o, states[i] | s' = 0)
                            where o is an observations.
        :return:            np array of shape (obs_num, S) where S is number of states
        """
        obs_num = features[0].shape[0]
        states_num = len(features)

        # dots is a list of values (f(o,s) . lambda) for different s values
        # logger.debug('features[0] = \n%s', two_d_arr_to_str(features[0]))
        # logger.debug('Lambda = %s', Lambda)
        # logger.debug('states_num = %s', states_num)
        dots = [np.squeeze(features[i].dot(Lambda)) for i in range(states_num)]
        # logger.debug('dots =\n%s', pprint.pformat(dots))

        TPM = np.zeros((obs_num, states_num))
        # logger.debug('TPM.shape = %s', TPM.shape)
        for i in range(states_num):
            exp_sum = np.sum(np.array([np.exp(dots[j] - dots[i]) for j in range(states_num) if j != i]), axis=0).T
            # logger.debug('exp_sum.shape = %s', exp_sum.shape)
            # logger.debug('exp_sum = %s', exp_sum)
            TPM[:, i] = 1 / (1 + exp_sum)

        return TPM

    def __build_expectation(self, features: list, TPM: np.ndarray):
        """
        :param features:  list of lists. features[i] contains the features of f(o, states[i] | s' = 0)
                            where o is an observations.
        :param TPM:      N * d array of TPM
        :return:        (d+1) array of expectation of features
        """
        states_num = len(features)
        E = np.sum(np.array([features[i].T.dot(TPM[:, i]) for i in range(states_num)]), axis=0)
        obs_num = features[0].shape[0]
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

    def decrease_dim(self, sequences):
        """
        Decrease dimensions of observations in sequences. Remove the dimensions related to the parents
        which has no activation (e.t. has no digit 1) in any observation.
        :param sequences:   list of sequences of (obs, state)
        :return:            new sequences, map of new indexes to the old ones; with one difference that
                            the observation is numpy array in tuple (obs, state)
        """
        dim = sequences[0][0][0].shape[1]
        stacked = np.vstack(list(reduce(lambda li1, li2: li1 + li2, [[pair[0] for pair in seq] for seq in sequences])))
        union = stacked.any(axis=0)
        orig_indexes = list(np.nonzero(union)[0])
        new_dim = len(orig_indexes)

        if new_dim == dim:
            return sequences, list(range(dim))

        # Decrease the dimensions and create the new sequences.
        new_sequences = [[(obs[:, orig_indexes], state) for obs, state in seq] for seq in sequences]

        return new_sequences, orig_indexes

    def sequences_to_feat_states(self, sequences):
        new_sequences, orig_indexes = self.decrease_dim(sequences)
        all_obs, state_mat = self.get_all_obs(new_sequences)
        features, C = self._calc_features(all_obs, state_mat)
        return features, state_mat


class LongMEMM(MEMM):
    def __init__(self, td_param=0.7):
        super().__init__()
        self.td_param = td_param

    def _calc_obs_features(self, obs, states):
        obs_dim = obs.shape[1]
        feat_dim = self.feat_dim if self.feat_dim else obs.shape[0] * obs_dim + 1
        if np.any(states):
            features = np.zeros((states.size, feat_dim - 1))
            mults = np.array([self.td_param ** i for i in range(obs.shape[0])])
            mults = np.tile(mults.reshape(obs.shape[0], 1), obs_dim)
            flat_obs = np.multiply(mults, obs).flatten()[:feat_dim - 1]
            nonzero_indexes = np.nonzero(states)[0]
            features[nonzero_indexes, :flat_obs.size] = np.tile(flat_obs, (nonzero_indexes.size, 1))
        else:
            features = np.zeros((states.size, feat_dim - 1))

        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((states.size, 1)) * C - np.reshape(feat_sum, (states.size, 1))
        features = np.concatenate((features, last_feat), axis=1)
        # logger.debug('features = \n%s', two_d_arr_to_str(features))
        return features, C

    def _calc_multi_obs_features(self, observations, states):
        if len(observations) != states.size:
            raise ValueError('number of observations and states must be equal')
        obs_num = len(observations)
        obs_dim = observations[0].shape[1]
        feat_dim = self.feat_dim if self.feat_dim else max(obs.shape[0] for obs in observations) * obs_dim + 1
        features = np.zeros((obs_num, feat_dim - 1))

        for ind in range(obs_num):
            if states[ind]:
                obs = observations[ind]
                mults = np.array([self.td_param ** i for i in range(obs.shape[0])])
                mults = np.tile(mults.reshape(obs.shape[0], 1), obs_dim)
                flat_obs = np.multiply(mults, obs).flatten()[:feat_dim - 1]
                # flat_obs = obs.flatten()[:feat_dim - 1]
                features[ind, :flat_obs.size] = flat_obs

        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        # logger.debug('features = \n%s', two_d_arr_to_str(features))
        return features, C


class BinMEMM(MEMM):
    def _calc_multi_obs_features(self, observations, states):
        if len(observations) != states.size:
            raise ValueError('number of observations and states must be equal')
        obs_num = len(observations)
        obs_dim = observations[0].shape[1]
        features = [np.any(obs, axis=0) for obs in observations]
        features = np.array(features)
        features = np.logical_and(features, np.tile(np.reshape(states, (obs_num, 1)), obs_dim))
        # features = np.logical_not(
        #     np.logical_xor(features, np.tile(np.reshape(states, (obs_num, 1)), obs_dim)))
        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        return features, C


class TDMEMM(MEMM):
    def __init__(self, td_param=0.7):
        super().__init__()
        self.td_param = td_param

    def _calc_multi_obs_features(self, observations, states):
        if len(observations) != states.size:
            raise ValueError('number of observations and states must be equal')
        obs_num = len(observations)
        obs_dim = observations[0].shape[1]
        features = []
        for obs in observations:
            mults = np.array([self.td_param ** i for i in range(obs.shape[0])])
            mults = np.tile(mults.reshape(obs.shape[0], 1), obs_dim)
            features.append(np.sum(np.multiply(mults, obs), axis=0))
        features = np.array(features)
        zero_state_indexes = np.where(np.logical_not(states))
        features[zero_state_indexes, :] = 1 - features[zero_state_indexes, :]
        # features[zero_state_indexes, :] = 0
        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        return features, C


class ParentTDMEMM(TDMEMM):
    def __init__(self, td_param=0.7):
        super().__init__()
        self.td_param = td_param

    def _calc_multi_obs_features(self, observations, states):
        if len(observations) != states.size:
            raise ValueError('number of observations and states must be equal')
        obs_num = len(observations)
        obs_dim = observations[0].shape[1]
        td_features, _ = super()._calc_features(observations, states != 0)
        # logger.debug('states = %s', states)
        features = np.zeros((obs_num, 2 * obs_dim))

        state_to_index = {self.orig_indexes[i] + 1: i for i in range(obs_dim)}
        # logger.debug('state_to_index = %s', state_to_index)
        for i in range(obs_num):
            s = int(states[i])
            if s == 0:
                features[i, :obs_dim] = td_features[i, :-1]
            elif s in state_to_index:
                ind = state_to_index[s]
                features[i, obs_dim + ind] = td_features[i, ind]

        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        # logger.debug('features = \n%s', two_d_arr_to_str(features[i, :]))
        return features, C


class LongParentTDMEMM(TDMEMM):
    def __init__(self, td_param=0.7):
        super().__init__()
        self.td_param = td_param

    def _calc_multi_obs_features(self, observations, states):
        if len(observations) != states.size:
            raise ValueError('number of observations and states must be equal')
        obs_num = len(observations)
        obs_dim = observations[0].shape[1]
        td_features, _ = super()._calc_features(observations, states != 0)
        # logger.debugv('obs_mat, state_mat =\n%s',
        #               np.concatenate((obs_mat, state_mat.reshape(state_mat.shape[0], 1)), axis=1))
        # logger.debugv('states =\n%s', states)
        features = np.zeros((obs_num, (obs_dim + 1) * obs_dim))
        # logger.debugv('features of state 0 update :\n%s', features)

        state_to_index = {self.orig_indexes[i] + 1: i for i in range(obs_dim)}
        # logger.debugv('state_to_index = %s', state_to_index)
        for i in range(obs_num):
            s = int(states[i])
            if s == 0:
                features[i, :obs_dim] = td_features[i, :-1]
            elif s in state_to_index:
                ind = state_to_index[s]
                start = (ind + 1) * obs_dim
                features[i, start: start + obs_dim] = td_features[i, :-1]

        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        # logger.debugv('features =\n%s', features)
        return features, C
