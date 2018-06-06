import copy
import logging
import numpy as np
import time
import scipy

logger = logging.getLogger('memm.memm')


class MEMM():
    def __init__(self):
        self.Lambda = {}
        self.TPM = None
        self.__all_obs = None
        self.__map_obs_index = {}
        #self.__map_index_obs = {}

    def fit(self, sequences, obs_dim):
        """
        Learn MEMM lambdas and transition probabilities for each previous state.
        :param sequences:   list of sequences. Each sequence is a list of tuples (observation, state).
                            Each observation is a string of 0 and 1 representing the activation state of input neighbors.
        :param states:      list of states
        :param obs_dim:     number of observation dimensions
        :return:            self
        """

        #t0 = time.time()
        new_sequences, map_index = self.__decrease_dim(sequences, obs_dim)
        #logger.info('time 1: %.2f', time.time() - t0)

        #t0 = time.time()
        all_obs_str = set()
        for seq in new_sequences:
            all_obs_str.update([obs for obs, _ in seq])
        all_obs_str = list(all_obs_str)
        self.__all_obs = [[int(d) for d in obs] for obs in all_obs_str]
        self.__all_obs = np.array(self.__all_obs, dtype=bool)
        self.__map_obs_index = {v: k for k, v in dict(enumerate(all_obs_str)).items()}
        #logger.info('time 2: %.2f', time.time() - t0)

        #t0 = time.time()
        epsilon = 0.1
        new_obs_dim = len(map_index)

        #logger.info('time 3: %.2f', time.time() - t0)
        #t0 = time.time()

        # Get tuples of (obs, state) which their previous state is 0.
        tuples, tuple_indexes = self.__divide_tuples(new_sequences, self.__map_obs_index)

        # Create observations and states matrices for tuples.
        obs_mat, state_mat = self.__create_matrices(tuples)

        # Calculate features for tuples observations/states. Shape of f1 is observations_num * (observations_dim+1)
        features, C = self.__calc_features(obs_mat, state_mat)

        # Calculate the training data average for each feature.
        F = np.mean(features, axis=0).T

        # Initialize Lambda as 1 then learn from training data.
        # Lambda is different per s' (previous state), But here just we use s' = 0.
        self.Lambda = np.ones(new_obs_dim + 1)

        #logger.info('time 4: %.2f', time.time() - t0)

        # GIS, run until convergence
        iter_count = 0
        while True:
            iter_count += 1
            #logger.info("iteration = %d ...", iter_count)
            Lambda0 = np.copy(self.Lambda)
            #t0 = time.time()
            self.TPM = self.__build_tpm(self.Lambda, self.__all_obs)
            #t1 = time.time() - t0
            #t0 = time.time()
            E = self.__build_expectation(obs_mat, self.TPM, tuple_indexes)
            #t2 = time.time() - t0
            #t0 = time.time()
            self.Lambda = self.__build_next_lambda(self.Lambda, C, F, E)
            #t3 = time.time() - t0
            #logger.info('iteration %d times: %.2f, %.2f, %.2f', iter_count, t1, t2, t3)

            if self.__check_lambda_convergence(Lambda0, self.Lambda, epsilon):
                logger.info('GIS iterations : %d', iter_count)
                break

        tpm_str = np.array2string(self.TPM, formatter={'float_kind': lambda x: "%.2f" % x})
        logger.info("TPM: \n%s", tpm_str[:100] + ' ...' if len(tpm_str) > 100 else tpm_str)
        lambda_str = np.array2string(self.Lambda, formatter={'float_kind': lambda x: "%.2f" % x})
        logger.info("lambda: %s", lambda_str[:100] + ' ...' if len(lambda_str) > 100 else lambda_str)

        # Increase dimensions of Lambda to the original ones.
        if obs_dim != new_obs_dim:
            self.Lambda = self.__inc_dimensions(self.Lambda, map_index, obs_dim)

        return self

    def predict(self, obs):
        """
        Predict the state conditioned on the given observation if the previous state is 0 (inactivated).
        :param obs:     current observation
        :return:        predicted next state
        """
        obs_vec = np.array([int(d) for d in obs], dtype=bool)
        if obs in self.__map_obs_index:
            index = self.__map_obs_index[obs]
        else:
            max_sim = -1
            index = -1
            for i in range(self.__all_obs.shape[0]):
                sim = np.sum(self.__all_obs[i, :] == obs_vec)
                if sim > max_sim:
                    max_sim = sim
                    index = i
        return 1 if self.TPM[index][1] > self.TPM[index][0] else 0

    def __create_matrices(self, tuples):
        obs_array = []
        state_array = []
        for obs, state in tuples:
            obs_array.append(obs)
            state_array.append(state)
        return np.array(obs_array, dtype=bool), np.array(state_array, dtype=bool)

    def __calc_features(self, obs_mat, state_mat):
        obs_num, obs_dim = obs_mat.shape
        f1 = np.multiply(obs_mat, np.tile(np.reshape(state_mat, (obs_num, 1)), obs_dim))
        feat_sum = np.sum(f1, axis=1)
        C = np.max(np.sum(obs_mat, axis=1)) + 1  # C is chosen so that is greater than sum of any row.
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        f1 = np.concatenate((f1, last_feat), axis=1)
        return f1, C

    def __divide_tuples(self, sequences, map_obs_index):
        """
        Return list of tuples of (obs, state) which their previous state is 0 (inactivated).
        :param sequences: list of sequences
        :return: list of tuples
        """
        tuples = []
        tuple_indexes = []

        for seq in sequences:
            if not seq:
                continue
            previous_state = seq[0][1]
            for obs, state in seq[1:]:
                if previous_state == 1:
                    break
                else:
                    obs_vec = np.array([int(d) for d in obs], dtype=bool)
                    tuples.append((obs_vec, state))
                    tuple_indexes.append(map_obs_index[obs])
                    previous_state = state

        return tuples, tuple_indexes

    def __build_tpm(self, Lambda, all_obs):
        """
        Create normalized transition probability matrix (TPM) from previous state of 0 (inactivated) given current observation
        :param Lambda:      np array of Lambda weights
        :param all_obs:     np array of all unique observations: obs_num * obs_dim
        :return:
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
        diff1 = np.reshape(scipy.special.expit(TPM[:, 0] - TPM[:, 1]), (obs_num, 1))
        diff2 = np.reshape(scipy.special.expit(TPM[:, 1] - TPM[:, 0]), (obs_num, 1))
        TPM = np.concatenate((diff1, diff2), axis=1)

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

    def __build_next_lambda(self, Lambda, C, F, E):
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

    def __check_lambda_convergence(self, Lambda0, Lambda1, epsilon):
        """
        Check if the lambdas are relatively the same.
        :param Lambda0: previous lambda
        :param Lambda1: current lambda
        :param epsilon: threshold of distance
        :return: True if the distance is lower than epsilon
        """
        return np.count_nonzero(np.absolute(Lambda0 - Lambda1) > epsilon) == 0

    def __decrease_dim(self, sequences, obs_dim):
        """
        Decrease dimensions of observations in sequences. Remove the dimensions related to the parents
        which has no activation (e.t. has no digit 1) in any observation.
        :param sequences:   list of sequences of (obs, state)
        :param obs_dim:     number of dimensions
        :return: new sequences, map of new indexes to the old ones
        """
        # Find the dimensions with any non-zero value.
        has_nonzero = [False for _ in range(obs_dim)]
        for i in range(len(sequences)):
            seq = sequences[i]
            for obs, state in seq:
                has_nonzero = [has_nonzero[j] or obs[j] == '1' for j in range(obs_dim)]

        if sum(has_nonzero) == obs_dim:
            return sequences, {i: i for i in range(obs_dim)}

        # Decrease the dimensions and create the new sequences.
        new_sequences = []
        for seq in sequences:
            new_sequences.append([])
            for obs, state in seq:
                new_obs = ''.join([obs[i] for i in range(obs_dim) if has_nonzero[i]])
                new_sequences[-1].append((new_obs, state))

        # Create map of new dimension indexes to the old indexes.
        map_index = {}
        i = 0
        for j in range(obs_dim):
            if has_nonzero[j]:
                map_index[i] = j
                i += 1

        return new_sequences, map_index

    def __inc_dimensions(self, Lambda, map_index, obs_dim):
        """
        Add the removed features to Lambda. Add all-zero dimensions in the correct positions.
        :param Lambda:      Lambda with decreased dimensions
        :param map_index:   map of indexes of decreased dimensions to the original indexes
        :param obs_dim:     original number of dimensions
        :return:            new Lambda
        """
        new_lambda = {i: float(0) for i in range(obs_dim)}
        for i in map_index:
            new_lambda[map_index[i]] = Lambda[i]
        return new_lambda
