import copy
import logging
import numpy

logger = logging.getLogger('memm.memm')


class MEMM():
    def __init__(self):
        self.Lambda = {}
        self.TPM = None
        self.__map_obs_index = {}
        self.__map_index_obs = {}

    def fit(self, sequences, obs_dim):
        """
        Learn MEMM lambdas and transition probabilities for each previous state.
        :param sequences:   list of sequences. Each sequence is a list of tuples (observation, state).
                            Each observation is a string of 0 and 1 representing the activation state of input neighbors.
        :param states:      list of states
        :param obs_dim:     number of observation dimensions
        :return:            self
        """

        new_sequences, map_index = self.__decrease_dim(sequences, obs_dim)

        all_obs = set()
        for seq in new_sequences:
            all_obs.update([obs for obs, state in seq])
        all_obs = list(all_obs)
        self.__map_obs_index = {v: k for k, v in dict(enumerate(all_obs)).items()}
        self.__map_index_obs = {v: k for k, v in self.__map_obs_index.items()}

        epsilon = 0.1
        new_obs_dim = len(map_index)
        C = new_obs_dim + 1                        # This should be number of features + 1
        self.TPM = self.__init_tpm(self.__map_index_obs)

        # Divide (o,s) into |S| buckets
        tuples = self.__divide_tuples(new_sequences)

        last_feature_list = self.__build_last_feature(new_obs_dim, C, self.__map_index_obs)

        # Initialize Lambda as 1 then learn from training data
        # Lambda is different per s' (previous state)
        F = self.__build_average_feature(tuples, new_obs_dim, last_feature_list)

        self.Lambda = self.__init_lambda(F)
        E = self.__init_expectation(F)

        # GIS, run until convergence
        iter_count = 0
        while True:
            iter_count += 1
            #logger.info("iteration = %d ...", iter_count)
            Lambda0 = copy.deepcopy(self.Lambda)
            self.__build_tpm(self.TPM, self.Lambda, new_obs_dim, self.__map_index_obs, last_feature_list)
            self.__build_expectation(E, tuples, new_obs_dim, last_feature_list, self.TPM, self.__map_obs_index)
            self.__build_next_lambda(self.Lambda, C, F, E)

            #tpm_str = str(self.TPM)
            #logger.info("TPM: %s", tpm_str[:200] + ' ...' if len(tpm_str) > 200 else '')
            #logger.info("lambda: %s", self.Lambda)
            if self.__check_lambda_convergence(Lambda0, self.Lambda, epsilon):
                logger.info('GIS iterations : %d', iter_count)
                break

        tpm_str = numpy.array2string(self.TPM, formatter={'float_kind': lambda x: "%.2f" % x})
        logger.info("TPM: %s", tpm_str[:100] + ' ...' if len(tpm_str) > 100 else tpm_str)
        logger.info("lambda: %s", self.Lambda)

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
        if obs not in self.__map_index_obs:
            return 0
        else:
            index = self.__map_obs_index[obs]
            return 1 if self.TPM[index][1] * 5 > self.TPM[index][0] else 0

    def __init_tpm(self, map_index_symbol):
        """
         Initialize TPM as all zeros
        """
        N = 2
        M = len(map_index_symbol)
        TPM = numpy.zeros(shape=(M, N), dtype=float)    # TPM (N * M) x N transitional probability matrix

        return TPM

    def __divide_tuples(self, sequences):
        """
        Return list of tuples of (obs, state) which their previous state is 0 (inactivated).
        :param sequences: list of sequences
        :return: list of tuples
        """
        tuples = []

        for seq in sequences:
            if not seq:
                continue
            previous_state = seq[0][1]
            for obs, state in seq[1:]:
                if previous_state == 1:
                    break
                else:
                    tuples.append((obs, state))
                    previous_state = state

        return tuples

    def __build_last_feature(self, obs_dim, C, map_index_obs):
        """
        Build the last, (o, s) specific feature.
        :param obs_dim: number of observation dimensions
        :param C: number of features (num of observations + 1)
        :param map_index_obs: map of indexes to observations
        :return: a dictionary that d[(obs, state)] = feature_value
        """
        last_feature_list = {}

        for state in [0, 1]:
            for j in map_index_obs:
                obs = map_index_obs[j]
                total = 0
                for l in range(obs_dim):
                    total += self.__feature(l, obs, state)
                last_feature_list[(obs, state)] = C - total

        return last_feature_list

    def __feature(self, l, obs, state, last_feature_list={}, obs_state_tuple=()):
        if l < len(obs):
            return 1 if obs[l] == '1' and state == 1 else 0
        else:
            temp_tuple = (obs, state)
            if temp_tuple == obs_state_tuple:
                return last_feature_list[(obs, state)] # This should error if last_feature_list is not passed in
            else:
                logger.info(temp_tuple)
                logger.info(obs_state_tuple)
                assert False, "Last feature error"

    def __build_average_feature(self, tuples, max_num_features, last_feature_list):
        """
        Calculate the training data average for each feature.
        Length = max_num_features + N * M
        :param tuples: list of (obs, state) tuples
        :param max_num_features: number of features
        :param last_feature_list: last dictionary of feature values
        :return: dictionary of feature numbers to average values
        """
        F = {}
        m_s = len(tuples)
        # Regular features + special, normalizing feature
        for l in range(max_num_features + 1):
            F[l] = float(0)
            if m_s == 0:
                continue
            for obs, state in tuples:
                obs_state_tuple = (obs, state)
                F[l] += self.__feature(l, obs, state, last_feature_list, obs_state_tuple)
            F[l] = F[l] / m_s

        return F

    def __init_lambda(self, F):
        """
        Initialize Lambda to 1's
        :param F: feature averages
        :return: lambda
        """
        Lambda = copy.deepcopy(F)
        for key in F:
            Lambda[key] = float(1)
        return Lambda

    def __init_expectation(self, F):
        """
        Initialize Expectation to 0's
        :param F: feature averages
        :return: expectations
        """
        E = copy.deepcopy(F)
        for key in F:
            E[key] = float(1)
        return E

    def __build_tpm(self, TPM, Lambda, max_num_features, map_index_obs, last_feature_list):
        """
        Create normalized transition probability matrix (TPM) from previous state of 0 (inactivated) given current observation
        :param TPM: current transition probability matrix
        :param Lambda: weights of features
        :param max_num_features: number of features
        :param map_index_obs: map of indexes to observations
        :param last_feature_list: feature values
        :return: new TPM
        """
        M = len(map_index_obs)

        # Calculate states
        for k in range(0, M):                # Current observation
            for state in [0, 1]:            # Current target state
                TPM[k][state] = float(0)
                obs = map_index_obs[k]
                obs_state_tuple = (obs, state)
                # Sum(Lambda_a * feature_a)
                for l in range(0, max_num_features + 1):  # Normal features + special feature
                    TPM[k][state] += Lambda[l] * self.__feature(l, obs, state, last_feature_list, obs_state_tuple)

                # Raise to exponential
                TPM[k][state] = numpy.exp(TPM[k][state])

            # Normalize
            row_sum = numpy.sum(TPM[k])
            for state in [0, 1]:            # Current target state
                TPM[k][state] = TPM[k][state] / row_sum

    def __build_expectation(self, E, tuples, max_num_features, last_feature_list, TPM, map_obs_index):
        """
        Calculate the expectation for each feature
        Note this is only for 1 bucket per call
        :param E: current expected value of features
        :param tuples: list of (obs, state) tuples
        :param max_num_features: number of features
        :param last_feature_list: feature values
        :param TPM: current TPM
        :param map_obs_index: map of indexes to observations
        :return: expected value of features
        """
        m_s = len(tuples)

        for l in range(max_num_features + 1):
            E[l] = float(0)
            if m_s == 0:
                continue
            for obs, state in tuples:
                k = map_obs_index[obs]
                for state in [0, 1]:
                    obs_state_tuple = (obs, state)
                    # logger.info(i, state, k, l)
                    E[l] += TPM[k][state] * self.__feature(l, obs, state, last_feature_list, obs_state_tuple)
            E[l] /= m_s

        return E

    def __build_next_lambda(self, Lambda, C, F, E):
        """
        Use Generalized iterative scaling (GIS) to learn Lambda parameter
        :param Lambda: current lambda (feature weights)
        :param C: number of features + 1
        :param F: feature averages
        :param E: expected value of features
        :return: next lambda
        """
        for feature_index in Lambda:
            assert not (E[feature_index] == 0 and E[feature_index] != F[feature_index]), \
                "E[{1}] == 0 but not F".format(feature_index)
            # If the average for the feature is 0, it has no contribution to the probability
            if F[feature_index] == 0:
                Lambda[feature_index] = 0
            else:
                log_F = numpy.log(F[feature_index])
                log_E = numpy.log(E[feature_index])
                Lambda[feature_index] = Lambda[feature_index] + (log_F - log_E) / C

    def __check_lambda_convergence(self, Lambda0, Lambda1, epsilon):
        """
        Check if the lambdas are relatively the same.
        :param Lambda0: previous lambda
        :param Lambda1: current lambda
        :param epsilon: threshold of distance
        :return: True if the distance is lower than epsilon
        """
        for feature_index in Lambda0:
            if abs(Lambda0[feature_index] - Lambda1[feature_index]) > epsilon:
                return False
        return True

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