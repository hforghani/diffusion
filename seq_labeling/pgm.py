import abc
import pprint
import typing
from functools import reduce
import numpy as np
import sklearn_crfsuite

from seq_labeling.utils import arr_to_str
from settings import logger
from utils.time_utils import time_measure


class SeqLabelModel(abc.ABC):
    def __init__(self):
        self.orig_indexes = []
        self.orig_indexes_map = {}

    @time_measure(level='debug')
    def fit(self, sequences: dict, iterations: int, states: list = None, **kwargs):
        """
        Learn MEMM lambdas and transition probabilities for previous state of 0.
        :param sequences:   list of sequences
        :param states:      list of all possible states
        :param iterations: maximum iterations
        :return:            self
        """
        new_sequences, self.orig_indexes = self.decrease_dim(sequences)
        new_dim = len(self.orig_indexes)
        self.orig_indexes_map = {self.orig_indexes[i]: i for i in range(len(self.orig_indexes))}

        if new_dim == 0:
            raise ValueError('Cannot train seq labeling model with all observations given zero')

        self._train(new_sequences, iterations, states, **kwargs)

        return self

    @abc.abstractmethod
    def _train(self, sequences, iterations, all_states=None, **kwargs):
        pass

    @abc.abstractmethod
    def get_prob(self, obs: np.ndarray, state: int, all_states: list, timers: list = None) -> float:
        """
        Get the probability of state=1 conditioned on the given observation and the previous state = 0 (inactivated).
        :param obs:     current observation
        :param state:   the state
        :param all_states:  list of all possible states
        """

    def get_all_obs(self, sequences):
        observations = []
        states = []
        for seq in sequences:
            for obs, state in seq:
                observations.append(obs)
                states.append(state)
        states = np.array(states)
        return observations, states

    def decrease_dim(self, sequences):
        """
        Decrease dimensions of observations in sequences. Remove the dimensions related to the parents
        which has no activation (e.t. has no digit 1) in any observation.
        :param sequences:   list of sequences of (obs, state)
        :return:            new sequences, map of new indexes to the old ones; with one difference that
                            the observation is numpy array in tuple (obs, state)
        """
        dim = sequences[0][0][0].size
        stacked = np.array(list(reduce(lambda li1, li2: li1 + li2, [[pair[0] for pair in seq] for seq in sequences])))
        union = stacked.any(axis=0)
        orig_indexes = list(np.nonzero(union)[0])
        new_dim = len(orig_indexes)

        if new_dim == dim:
            return sequences, list(range(dim))

        # Decrease the dimensions and create the new sequences.
        new_sequences = [[(obs[orig_indexes], state) for obs, state in seq] for seq in sequences]

        return new_sequences, orig_indexes


class MEMM(SeqLabelModel, abc.ABC):
    def __init__(self):
        super().__init__()
        self.Lambda = None
        self.feat_dim = None

    def set_params(self, Lambda, orig_indexes):
        self.Lambda = Lambda
        self.orig_indexes = orig_indexes
        self.orig_indexes_map = {self.orig_indexes[i]: i for i in range(len(self.orig_indexes))}
        self.feat_dim = Lambda.size
        logger.debug('self.feat_dim set to %d', self.feat_dim)

    def _train(self, sequences, iterations, all_states=None, **kwargs):
        # Get all observations and states of which previous state is zero.
        all_obs, state_mat = self.get_all_obs(sequences)
        logger.debugv('all_obs =\n%s', all_obs)
        logger.debugv('state_mat =\n%s', state_mat)
        logger.debugv('all possible states = %s', all_states)

        # Calculate features for observation-state pairs. Shape of features is feat_num * feat_dim.
        features, C = self._calc_multi_seq_features(sequences)
        self.feat_dim = features.shape[1]
        logger.debugv('features =\n%s', features)
        features_for_all_states = [self._calc_multi_seq_features([[(obs, s) for obs, _ in seq] for seq in sequences])[0]
                                   for s in all_states]
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

    def get_prob(self, obs_seq, state, all_states, timers=None):
        # if timers is None:
        #     timers = [Timer(f'get_prob part {i}', level='debug', unit=TimeUnit.SECONDS) for i in range(2)]

        obs_seq = [obs[:, self.orig_indexes] for obs in obs_seq]
        if all(np.count_nonzero(obs) == 0 for obs in obs_seq):
            return 0
        f, _ = self._calc_seq_features(obs_seq, np.array(all_states))
        dots = f.dot(self.Lambda.reshape(self.Lambda.shape[0], 1))
        s_value = dots[all_states.index(state)]
        prob = float(1 / np.sum(np.array([np.exp(dots[i] - s_value) for i in range(len(all_states))])))
        return prob

    def get_probs(self, obs_seq: list, all_states: list) -> list:
        """
        Get the list of probabilities of transition to all states given the observation conditioned on that the previous
        state is inactive.
        :param obs_seq:     sequence of t observations
        :param all_states:  list of all possible states
        :return:            list of probabilities of P(Y_t=1|X,Y_{t-1}=0) related to the states given
        """
        obs_seq = [obs[self.orig_indexes] for obs in obs_seq]
        f, _ = self._calc_seq_features(obs_seq, np.array(all_states))
        dots = f.dot(self.Lambda.reshape(self.Lambda.shape[0], 1))
        probs = np.zeros(len(all_states))
        for i in range(len(all_states)):
            exp_sum = np.sum(np.array([np.exp(dots[j] - dots[i]) for j in range(len(all_states)) if j != i]))
            probs[i] = 1 / (1 + exp_sum)
        return probs.tolist()

    def get_multi_obs_probs(self, sequences: list, all_states: list) -> np.ndarray:
        probs = []
        for seq in sequences:
            obs_seq = [obs for obs, state in seq]
            for i in range(1, len(seq)):
                probs.append(np.array(self.get_probs(obs_seq[:i], all_states)))
        probs = np.array(probs)
        # logger.debug('sequences = \n%s', sequences)
        # logger.debug('probs = \n%s', probs)
        return probs

    def _calc_seq_features(self, obs_seq: list, states: np.ndarray) -> np.ndarray:
        sequences = [[(obs, state) for obs in obs_seq] for state in states]
        return self._calc_multi_seq_features(sequences, only_last=True)

    @abc.abstractmethod
    def _calc_multi_seq_features(self, sequences: list, only_last: bool = False) -> np.ndarray:
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
        # logger.debug('features[0] = \n%s', arr_to_str(features[0]))
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

    def sequences_to_feat_states(self, sequences):
        new_sequences, orig_indexes = self.decrease_dim(sequences)
        features, C = self._calc_multi_seq_features(new_sequences)
        return features


class LongMEMM(MEMM):

    def _calc_seq_features(self, obs_seq, states):
        obs_dim = obs_seq[0].size
        D = (self.feat_dim - 1) // obs_dim
        features = np.zeros((states.size, obs_dim * D))

        if np.any(states):
            feat = np.concatenate(obs_seq[::-1], axis=None) if len(obs_seq) > 1 else obs_seq[0].copy()
            feat.resize(obs_dim * D)
            nonzero_indexes = np.nonzero(states)[0]
            features[nonzero_indexes, :] = np.tile(feat, (nonzero_indexes.size, 1))

        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((states.size, 1)) * C - np.reshape(feat_sum, (states.size, 1))
        features = np.concatenate((features, last_feat), axis=1)
        # logger.debug('features = \n%s', two_d_arr_to_str(features))
        return features, C

    def _calc_multi_seq_features(self, sequences, only_last=False):
        obs_dim = sequences[0][0][0].size
        D = (self.feat_dim - 1) // obs_dim if self.feat_dim else max(len(seq) for seq in sequences)
        features = []

        for seq in sequences:
            if only_last:
                state = seq[-1][1]
                if state:
                    feat = np.concatenate([obs for obs, state in seq[::-1]], axis=None) if len(seq) > 1 else seq[
                        0].copy()
                else:
                    feat = np.zeros(D * obs_dim)
                np.resize(D * obs_dim)
                features.append(feat)
            else:
                if any(state for obs, state in seq):
                    obs_seq = np.array([])
                    for obs, state in seq:
                        obs_seq = np.concatenate((obs, obs_seq), axis=None)
                        feat = obs_seq.copy() if state else np.zeros(D * obs_dim)
                        feat.resize(D * obs_dim)
                        features.append(feat)
                else:
                    features.extend([np.zeros(D * obs_dim)] * len(seq))
        features = np.array(features)

        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        feat_count = features.shape[0]
        last_feat = np.ones((feat_count, 1)) * C - np.reshape(feat_sum, (feat_count, 1))
        features = np.concatenate((features, last_feat), axis=1)
        # logger.debug('features = \n%s', two_d_arr_to_str(features))
        return features, C


class BinMEMM(MEMM):
    def _calc_seq_features(self, obs_seq, states):
        obs_dim = obs_seq[0].size
        features = np.zeros((states.size, obs_dim))

        if np.any(states):
            feat = np.any(np.array(obs_seq), axis=0)
            nonzero_indexes = np.nonzero(states)[0]
            features[nonzero_indexes, :] = np.tile(feat, (nonzero_indexes.size, 1))

        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((states.size, 1)) * C - np.reshape(feat_sum, (states.size, 1))
        features = np.concatenate((features, last_feat), axis=1)
        return features, C

    def _calc_multi_seq_features(self, sequences, only_last=False):
        obs_dim = sequences[0][0][0].size
        features = []

        for seq in sequences:
            if only_last:
                state = seq[-1][1]
                feat = np.any(np.array([obs for obs, state in seq]), axis=0) if state else np.zeros(obs_dim)
                features.append(feat)
            else:
                if any(state for obs, state in seq):
                    obs_sum = np.zeros(obs_dim)
                    for obs, state in seq:
                        obs_sum += obs
                        feat = obs_sum.copy() if state else np.zeros(obs_dim)
                        features.append(feat)
                else:
                    features.extend([np.zeros(obs_dim)] * len(seq))
        features = np.array(features)

        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        feat_count = features.shape[0]
        last_feat = np.ones((feat_count, 1)) * C - np.reshape(feat_sum, (feat_count, 1))
        features = np.concatenate((features, last_feat), axis=1)
        return features, C


class TDMEMM(MEMM):
    def __init__(self, td_param=0.5):
        super().__init__()
        self.td_param = td_param

    def _calc_seq_features(self, obs_seq, states):
        # TODO
        obs_dim = obs_seq.shape[1]
        features = np.zeros((states.size, obs_dim))

        if np.any(states):
            mults = np.array([self.td_param ** i for i in range(obs_seq.shape[0])])
            mults = np.tile(mults.reshape(obs_seq.shape[0], 1), obs_dim)
            feat = np.sum(np.multiply(mults, obs_seq), axis=0)
            nonzero_indexes = np.nonzero(states)[0]
            features[nonzero_indexes, :] = np.tile(feat, (nonzero_indexes.size, 1))

        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((states.size, 1)) * C - np.reshape(feat_sum, (states.size, 1))
        features = np.concatenate((features, last_feat), axis=1)
        return features, C

    def _calc_multi_seq_features(self, sequences, only_last=False):
        # TODO
        if len(observations) != states.size:
            raise ValueError('number of observations and states must be equal')
        obs_num = len(observations)
        obs_dim = observations[0].shape[1]
        features = np.zeros((obs_num, obs_dim))

        for ind in range(obs_num):
            if states[ind]:
                obs = observations[ind]
                mults = np.array([self.td_param ** i for i in range(obs.shape[0])])
                mults = np.tile(mults.reshape(obs.shape[0], 1), obs_dim)
                features[ind, :] = np.sum(np.multiply(mults, obs), axis=0)

        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        return features, C


class ParentTDMEMM(TDMEMM):
    def __init__(self, td_param=0.5):
        super().__init__()
        self.td_param = td_param

    def _calc_multi_seq_features(self, sequences, only_last=False):
        # TODO
        if len(observations) != states.size:
            raise ValueError('number of observations and states must be equal')
        obs_num = len(observations)
        obs_dim = observations[0].shape[1]
        td_features, _ = super()._calc_multi_seq_features(observations, states != 0)
        # for obs in observations:
        #     logger.debug('obs = \n%s', obs_to_str(obs))
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
        # logger.debug('features = \n%s', two_d_arr_to_str(features))
        return features, C


class LongParentTDMEMM(TDMEMM):
    def __init__(self, td_param=0.5):
        super().__init__()
        self.td_param = td_param

    def _calc_multi_seq_features(self, sequences, only_last=False):
        # TODO
        if len(observations) != states.size:
            raise ValueError('number of observations and states must be equal')
        obs_num = len(observations)
        obs_dim = observations[0].shape[1]
        td_features, _ = super()._calc_multi_seq_features(observations, states != 0)
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


class CRF(SeqLabelModel):
    def _train(self, sequences, iterations, all_states=None, **kwargs):
        alg = kwargs.get('algorithm', 'lbfgs')
        params = {
            'lbfgs': ['c1', 'c2', 'linesearch'],
            'l2sgd': ['c2', 'calibration_rate'],
            'ap': [],
            'pa': ['pa_type', 'c', 'error_sensitive', 'averaging'],
            'arow': ['variance', 'gamma'],
        }
        crf_params = {param: kwargs[param] for param in params[alg] if param in kwargs}
        crf = sklearn_crfsuite.CRF(algorithm=alg, max_iterations=iterations, **crf_params)

        features = [self._obs_feat_sequence([obs for obs, state in seq]) for seq in sequences]
        states = [[self.__state_to_str(0)] + [self.__state_to_str(state) for obs, state in seq] for seq in sequences]
        logger.debugv('sequences = \n%s', pprint.pformat(sequences))
        logger.debugv('features = \n%s', pprint.pformat(features))
        logger.debugv('states = \n%s', pprint.pformat(states))
        # logger.debugv('features & states = \n%s', pprint.pformat([list(zip(f, s)) for f, s in zip(features, states)]))

        crf.fit(features, states)
        logger.debugv('attributes_:\n%s', crf.attributes_)
        logger.debugv('state_features_:\n%s', pprint.pformat(crf.state_features_))
        logger.debugv('transition_features_:\n%s', pprint.pformat(crf.transition_features_))

        self.crf = crf

    def __state_to_str(self, state):
        if isinstance(state, bool):
            return '1' if state else '0'
        else:
            return str(state)

    def _obs_feat_sequence(self, obs_seq):
        logger.debugv('_obs_feat_sequence -> obs_seq = %s', obs_seq)
        dim = obs_seq[0].size
        zero_obs = np.zeros(dim, dtype=bool)
        zero_feat = self._obs_feature([zero_obs])
        return [zero_feat] + [self._obs_feature(obs_seq[:i + 1]) for i in range(len(obs_seq))]

    def _obs_feature(self, obs_seq):
        logger.debugv('_obs_feature -> obs_seq = %s', obs_seq)
        return {f'{-len(obs_seq) + 1 + i}:{j}': obs_seq[i][j] for i in range(len(obs_seq)) for j in
                range(obs_seq[i].size)}

    def get_prob(self, obs_seq, state, all_states, timers=None):
        # if timers is None:
        #     timers = [Timer(f'get_prob part {i}', level='debug', unit=TimeUnit.SECONDS) for i in range(2)]

        features = self._obs_feat_sequence(obs_seq)
        probs = self.crf.predict_marginals_single(features)
        logger.debugv('features = \n%s', features)
        logger.debugv('probs = \n%s', probs)
        return probs[-1][self.__state_to_str(state)]

    def get_probs(self, obs_seq: np.ndarray, all_states: list) -> list:
        """
        Get the list of probabilities of transition to all states given the observation conditioned on that the previous
        state is inactive.
        :param obs_seq:         observation sequence
        :param all_states:  list of all possible states
        :return:            list of probabilities related to the states given
        """
        logger.debugv('obs_seq = \n%s', obs_seq)
        features = self._obs_feat_sequence(obs_seq)
        probs = self.crf.predict_marginals_single(features)
        logger.debugv('features = \n%s', features)
        logger.debugv('probs = \n%s', probs)
        probs_list = [probs[-1].get(self.__state_to_str(state), 0) for state in all_states]
        logger.debugv('probs_list = \n%s', probs_list)
        return probs_list


class BinCRF(CRF):
    def _obs_feature(self, obs_seq):
        logger.debugv('_obs_feature -> obs_seq = %s', obs_seq)
        bin_feat = np.any(np.array(obs_seq), axis=0)
        f = {str(i): bin_feat[i] for i in range(bin_feat.size)}
        logger.debugv('f = %s', f)
        return f


class TDCRF(CRF):
    def __init__(self, td_param=0.5):
        super().__init__()
        self.td_param = td_param

    def _obs_feature(self, obs_seq):
        #TODO
        mults = np.array([self.td_param ** i for i in range(obs_seq.shape[0])])
        mults = np.tile(mults.reshape(obs_seq.shape[0], 1), obs_seq.shape[1])
        td_feat = np.sum(np.multiply(mults, obs_seq), axis=0)
        return {str(i): td_feat[i] for i in range(td_feat.size)}
