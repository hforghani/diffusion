class Metric(object):
    def __init__(self, output, true_output, ref=None):
        """
        :param output:          items predicted by the system to be positive
        :param true_output:     real positive items
        :param ref:         all positive and negative items
        """
        self.output = output
        self.true_output = true_output
        out_set = set(output)
        true_set = set(true_output)

        if ref is None:
            self.ref = out_set | true_set
        else:
            self.ref = set(ref)

        self.tp = len(out_set.intersection(true_set))
        self.fp = len(out_set - true_set)
        self.fn = len(true_set - out_set)
        self.tn = len(self.ref - out_set - true_set)

        self.__precision = None
        self.__recall = None
        self.__f1 = None
        self.__fpr = None

    def precision(self):
        if self.__precision is None:
            self.__precision = self.tp / len(self.output) if self.output else 1
        return self.__precision

    def recall(self):
        if self.__recall is None:
            self.__recall = self.tp / len(self.true_output) if self.true_output else 1
        return self.__recall

    def f1(self):
        if self.__f1 is None:
            self.__f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn) if self.tp != 0 else 0
        return self.__f1

    def fpr(self):
        """
        Calculate false positive rate.
        :return:
        """
        if self.__fpr is None:
            self.__fpr = self.fp / (self.fp + self.tn)
        return self.__fpr

    def prp(self, prob):
        """
        Calculate Precision at different Recall Points.
        param prob: dictionary of mapping from outputs to their probabilities
        """
        sorted_output = sorted(self.output, key=lambda val: prob[val], reverse=True)
        prp_val = []
        for i in range(len(sorted_output)):
            if sorted_output[i] in self.true_output:
                prp_val.append(Metric(sorted_output[:i + 1], self.true_output).precision())
        return prp_val

    def set(self, precision, recall, f1, fpr):
        self.__precision = precision
        self.__recall = recall
        self.__f1 = f1
        self.__fpr = fpr
