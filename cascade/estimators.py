import numpy as np
from sklearn.base import BaseEstimator

from settings import logger


class MEMMEstimator(BaseEstimator):
    def __init__(self, memm, threshold=None):
        self.memm = memm
        self.threshold = threshold

    def fit(self, observations, states):
        return self

    def predict(self, observations):
        probs = self.memm.multi_prob(observations)
        states = probs >= self.threshold
        return states
