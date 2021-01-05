import numpy as np
import scipy
import time

times = [0] * 15


class MemmException(Exception):
    pass


def obs_to_array(obs, dim):
    obs_copy = obs
    obs_arr = []
    for _ in range(dim):
        obs_arr.append(obs_copy % 2)
        obs_copy >>= 1
    obs_arr = obs_arr[::-1]
    return np.array(obs_arr, dtype=int)


def array_to_obs(arr):
    obs = 0
    for d in arr:
        obs <<= 1
        obs += int(d)
    return obs


class MEMM():
    def __init__(self):
        self.Lambda = None
        self.TPM = None
        self.all_obs_arr = None
        self.map_obs_index = {}
        self.orig_indexes = []

    def fit(self, evidence):
        """
        Learn MEMM lambdas and transition probabilities for each previous state.
        :param evidence:   an instance of MemmEvidence
        :return:            self
        """
        t0 = t_start = time.time()
        dim, sequences = evidence[0], evidence[1]
        new_sequences, self.orig_indexes = self.__decrease_dim(sequences, dim)
        new_dim = len(self.orig_indexes)
        times[1] += time.time() - t0
        #logger.info('time 1: %.2f', time.time() - t0)

        if new_dim == 0:
            raise MemmException('Cannot train MEMM with all observations given zero')

        t0 = time.time()
        all_obs = set()
        for seq in new_sequences:
            all_obs.update([pair[0] for pair in seq])
        all_obs = list(all_obs)
        self.all_obs_arr = [obs_to_array(obs, new_dim) for obs in all_obs]
        self.all_obs_arr = np.array(self.all_obs_arr)
        self.map_obs_index = {v: k for k, v in dict(enumerate(all_obs)).items()}
        times[2] += time.time() - t0
        #logger.info('time 2: %.2f', time.time() - t0)

        t0 = time.time()
        epsilon = 0.1

        # Get pairs of (obs, state) which their previous state is 0.
        rel_pairs, rel_indexes = self.__get_related_pairs(new_sequences, self.map_obs_index)

        #logger.info('time 3: %.2f', time.time() - t0)
        times[3] += time.time() - t0
        t0 = time.time()

        # Create observations and states matrices for related pairs.
        obs_mat, state_mat = self.__create_matrices(rel_pairs, rel_indexes, new_dim)

        times[13] += time.time() - t0
        t0 = time.time()

        # Calculate features for observation-state pairs. Shape of f1 is obs_num * (obs_dim+1)
        features, C = self.__calc_features(obs_mat, state_mat)

        # Calculate the training data average for each feature.
        F = np.mean(features, axis=0).T

        # Initialize Lambda as 1 then learn from training data.
        # Lambda is different per s' (previous state), But here just we use s' = 0.
        self.Lambda = np.ones(new_dim + 1)

        #logger.info('time 4: %.2f', time.time() - t0)
        times[4] += time.time() - t0
        t0 = time.time()

        # GIS, run until convergence
        iter_count = 0
        nnz_list = [] # for test
        while True:
            iter_count += 1
            #logger.info("iteration = %d ...", iter_count)
            Lambda0 = np.copy(self.Lambda)
            t = time.time()
            self.TPM = self.__build_tpm(self.Lambda, self.all_obs_arr)
            times[9] += time.time() - t
            t = time.time()
            E = self.__build_expectation(obs_mat, self.TPM, rel_indexes)
            times[10] += time.time() - t
            t = time.time()
            self.Lambda = self.__build_next_lambda(self.Lambda, C, F, E)
            times[11] += time.time() - t

            t = time.time()
            #if self.__check_lambda_convergence(Lambda0, self.Lambda, epsilon):
            nnz = np.count_nonzero(np.absolute(Lambda0 - self.Lambda) > epsilon)
            nnz_list.append(nnz)
            times[12] += time.time() - t
            if nnz == 0:
                #if log > 0:
                #    logger.info('distances in iterations: %s', nnz_list)
                #    logger.info('GIS iterations : %d', iter_count)
                break

        #logger.info('time 5: %.2f', time.time() - t0)
        times[5] += time.time() - t0
        times[0] += time.time() - t_start

        return self

    def predict(self, obs, dim, threshold=None):
        """
        Predict the state conditioned on the given observation if the previous state is 0 (inactivated).
        :param obs:     current observation
        :param dim:     dimension of observation vector
        :return:        predicted next state
        """
        #logger.debug('running MEMM predict method ...')
        new_obs = self.__decrease_dim_by_indexes(obs, dim, self.orig_indexes)
        new_dim = len(self.orig_indexes)
        if new_obs in self.map_obs_index:
            index = self.map_obs_index[new_obs]
            #logger.debug('obs found. prob = %f', self.TPM[index][1])
        else:
            #return 0
            new_obs_vec = obs_to_array(new_obs, new_dim)
            obs_num = self.all_obs_arr.shape[0]
            sim = np.sum(self.all_obs_arr == np.tile(new_obs_vec, (obs_num, 1)), axis=1)
            index = np.argmax(sim)
            #logger.debug('obs not found. sim = %f. prob = %f', np.max(sim) / new_dim, self.TPM[index][1])
        if threshold is None:
            threshold = 0.5

        next_state = 1 if self.TPM[index][1] >= threshold else 0
        return next_state

    def __create_matrices(self, pairs, indexes, dim):
        obs_array = [self.all_obs_arr[indexes[i], :] for i in range(len(pairs))]
        state_array = [pair[1] for pair in pairs]
        return np.array(obs_array, dtype=bool), np.array(state_array, dtype=bool)

    def __calc_features(self, obs_mat, state_mat):
        obs_num, obs_dim = obs_mat.shape
        #features = np.logical_and(obs_mat, np.tile(np.reshape(state_mat, (obs_num, 1)), obs_dim))
        features = np.logical_not(np.logical_xor(obs_mat, np.tile(np.reshape(state_mat, (obs_num, 1)), obs_dim)))
        #C = np.max(np.sum(obs_mat, axis=1)) + 1  # C is chosen so that is greater than sum of any row.
        C = obs_dim + 1
        feat_sum = np.sum(features, axis=1)
        last_feat = np.ones((obs_num, 1)) * C - np.reshape(feat_sum, (obs_num, 1))
        features = np.concatenate((features, last_feat), axis=1)
        return features, C

    def __get_related_pairs(self, sequences, map_obs_index):
        """
        Return related MemmObsPair's means the ones which their previous state is 0 (inactivated).
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

    def __decrease_dim(self, sequences, dim):
        """
        Decrease dimensions of observations in sequences. Remove the dimensions related to the parents
        which has no activation (e.t. has no digit 1) in any observation.
        :param sequences:   list of sequences of (obs, state)
        :param dim:         number of observation dimensions
        :return:            new sequences, map of new indexes to the old ones; with one difference that
                            the observation is numpy array in tuple (obs, state)
        """
        # Find the dimensions with any non-zero value.
        t0 = time.time()
        has_nonzero = 0
        for seq in sequences:
            for obs, state in seq[1:]:
                has_nonzero |= obs
        times[6] += time.time() - t0

        # Create map of new dimension indexes to the old indexes.
        t0 = time.time()
        orig_indexes = []
        has_nnz_copy = has_nonzero
        for i in range(dim):
            if has_nnz_copy % 2:
                orig_indexes.append(dim - i - 1)
            has_nnz_copy >>= 1
        orig_indexes.sort()
        times[7] += time.time() - t0
        #logger.info('orig_indexes = %s', orig_indexes)

        # Count the used (nonzero) dimensions
        new_dim = len(orig_indexes)

        #logger.info('dim : %d, new dim: %d', dim, new_dim)
        if new_dim == dim:
            return sequences, {i: i for i in range(dim)}

        # Decrease the dimensions and create the new sequences.
        t0 = time.time()
        new_sequences = []
        for seq in sequences:
            #logger.info('sequence:')
            new_seq = []
            for obs, state in seq:
                new_obs = self.__decrease_dim_by_indexes(obs, dim, orig_indexes)
                new_seq.append((new_obs, state))
                #logger.info('obs: %d', obs)
                #logger.info('obs: %s', obs_to_array(obs, dim).astype(int))
                #logger.info('new obs: %d', new_obs)
                #logger.info('%s : %d', obs_to_array(new_obs, new_dim).astype(int), state)
            new_sequences.append(new_seq)
        times[8] += time.time() - t0

        return new_sequences, orig_indexes

    def __decrease_dim_by_indexes(self, obs, dim, orig_indexes):
        new_obs = 0
        for ind in orig_indexes:
            new_obs <<= 1
            new_obs += (obs >> (dim - ind - 1)) % 2
        return new_obs

    def __inc_matrix_dim(self, matrix, orig_indexes, obs_dim):
        """
        Add the removed features to matrix (observations or lambda). Add all-zero dimensions in the correct positions.
        :param matrix:          if a vector with size D is given it increases its dimension to obs_dim using map_index.
                                if a matrix with size N*D is given it returns a matrix with size N*obs_dim using map_index.
        :param orig_indexes:    list of indexes of original indexes related to each decreased dimension
        :param obs_dim:         original number of dimensions
        :return:                the matrix with original dimensions
        """
        if len(matrix.shape) == 1:
            orig_matrix = np.zeros(obs_dim, dtype=matrix.dtype)
            orig_matrix[orig_indexes] = matrix
        else:
            orig_matrix = np.zeros((matrix.shape[0], obs_dim), dtype=matrix.dtype)
            orig_matrix[:, orig_indexes] = matrix
        return orig_matrix

    def __inc_map_obs_dim(self, map_obs_index, orig_indexes, obs_dim):
        new_map = {}
        new_dim = len(orig_indexes)
        for obs, index in map_obs_index.items():
            obs_vec = obs_to_array(obs, new_dim)
            orig_vec = np.zeros(obs_dim, dtype=bool)
            orig_vec[orig_indexes] = obs_vec
            orig_obs = array_to_obs(orig_vec)
            new_map[orig_obs] = index
            #logger.info('obs: %d', obs)
            #logger.info('obs: %s', obs_vec.astype(int))
            #logger.info('orig indexes: %s', orig_indexes)
            #logger.info('orig obs: %s', obs_to_array(orig_obs, obs_dim).astype(int))
        return new_map
