class Validation(object):
    def __init__(self, output, true_output):
        self.output = list(output)
        self.true_output = list(true_output)

    def precision(self):
        if len(self.output) == 0:
            return 1
        tp = len(set(self.output).intersection(set(self.true_output)))
        return float(tp) / len(self.output)

    def recall(self):
        if len(self.true_output) == 0:
            return 1
        tp = len(set(self.output).intersection(set(self.true_output)))
        return float(tp) / len(self.true_output)

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
