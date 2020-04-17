class MemmEvidence:
    def __init__(self, sequences, dim):
        self.sequences = sequences # list of sequences of MemmObsPair.
        self.dim = dim # observation dimension.  e.g dimension of binary representation 101 is 3


class MemmObsPair:
    def __init__(self, obs, state):
        self.obs = obs # observation in binary presentation. e.g 5 = 101
        self.state = state # activeness state including 0 or 1