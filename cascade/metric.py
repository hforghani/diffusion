class Metric(object):
    def __init__(self, output, true_output, ref=None):
        """
        :param output:          items predicted by the system to be positive
        :param true_output:     real positive items
        :param ref:         all positive and negative items
        """
        out_set = set(output)
        true_set = set(true_output)
        ref = set(ref) if ref is not None else out_set | true_set

        tp = len(out_set.intersection(true_set))
        fp = len(out_set - true_set)
        fn = len(true_set - out_set)
        tn = len(ref - out_set - true_set)

        self.metrics = {
            "precision": self.__precision(tp, output),
            "recall": self.__recall(tp, true_output),
            "f1": self.__f1(tp, fp, fn),
            "fpr": self.__fpr(fp, tn),
            "accuracy": self.__accuracy(tp, fp, fn, tn),
        }

    def __precision(self, tp, output):
        return tp / len(output) if output else 1

    def __recall(self, tp, true_output):
        return tp / len(true_output) if true_output else 1

    def __f1(self, tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if tp != 0 else 0

    def __fpr(self, fp, tn):
        """
        Calculate false positive rate.
        :return:
        """
        return fp / (fp + tn) if fp + tn != 0 else 0

    def __accuracy(self, tp, fp, fn, tn):
        return (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0

    def __prp(self, prob, output, true_output):
        """
        Calculate Precision at different Recall Points.
        param prob: dictionary of mapping from outputs to their probabilities
        """
        sorted_output = sorted(output, key=lambda val: prob[val], reverse=True)
        prp_val = []
        for i in range(len(sorted_output)):
            if sorted_output[i] in true_output:
                prp_val.append(Metric(sorted_output[:i + 1], true_output).precision)
        return prp_val

    @staticmethod
    def from_values(**kwargs):
        metric = Metric([], [])
        metric.metrics = kwargs
        return metric

    def __getitem__(self, item):
        if item in self.metrics:
            return self.metrics[item]
        else:
            raise ValueError(f"metric `{item}` has not been set")
