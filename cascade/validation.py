class Validation(object):
    def __init__(self, output, true_output):
        self.output = list(output)
        self.true_output = list(true_output)
        self.tp = len(set(self.output).intersection(set(self.true_output)))
        self.fp = len(set(self.output) - set(self.true_output))
        self.fn = len(set(self.true_output) - set(self.output))

    def precision(self):
        if len(self.output) == 0:
            return 1
        return float(self.tp) / len(self.output)

    def recall(self):
        if len(self.true_output) == 0:
            return 1
        return float(self.tp) / len(self.true_output)

    def f1(self):
        return 2 * float(self.tp) / (2 * self.tp + self.fp + self.fn)

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
