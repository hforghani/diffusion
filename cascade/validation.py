class Validation(object):
    def __init__(self, output, true_output, ref=None):
        """
        :param output:          items predicted by the system to be positive
        :param true_output:     real positive items
        :param ref:         all positive and negative items
        """
        self.output = list(output)
        self.true_output = list(true_output)
        out_set = set(output)
        true_set = set(true_output)

        if ref is None:
            ref_set = out_set.union(true_set)
            self.ref = list(ref_set)
        else:
            ref_set = set(ref)
            self.ref = list(ref)

        self.tp = len(out_set.intersection(true_set))
        self.fp = len(out_set - true_set)
        self.fn = len(true_set - out_set)
        self.tn = len(ref_set - out_set - true_set)

    def precision(self):
        if len(self.output) == 0:
            return 1
        return float(self.tp) / len(self.output)

    def recall(self):
        if len(self.true_output) == 0:
            return 1
        return float(self.tp) / len(self.true_output)

    def f1(self):
        if self.tp == 0:
            return 0
        return 2 * float(self.tp) / (2 * self.tp + self.fp + self.fn)

    def fpr(self):
        """
        Calculate false positive rate.
        :return:
        """
        return self.fp / (self.fp + self.tn)

    def prp(self, prob):
        """
        Calculate Precision at different Recall Points.
        param prob: dictionary of mapping from outputs to their probabilities
        """
        sorted_output = sorted(self.output, key=lambda val: prob[val], reverse=True)
        prp_val = []
        for i in range(len(sorted_output)):
            if sorted_output[i] in self.true_output:
                prp_val.append(Validation(sorted_output[:i + 1], self.true_output).precision())
        return prp_val
