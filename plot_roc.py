import argparse
import itertools
import json
import os
import re
from typing import Dict

from matplotlib import pyplot
import numpy as np

from diffusion.enum import Criterion

METHODS = {"aslt": "AsLT",
           "emic": "EMIC",
           "ctic": "CTIC",
           "daic": "DAIC",
           "infvae": "Inf-VAE",
           "deepdiffuse": "DeepDiffuse",
           "mlmemm": "LPMEMM",
           "mbmemm": "SPMEMM",
           "mlcrf": "LPCRF",
           "mbcrf": "SPCRF"}


def plot_roc(data: Dict[str, Dict[str, np.array]]):
    """
    Save ROC plot as png and FPR-TPR values as json.
    """
    pyplot.figure()
    line_styles = itertools.cycle(["-", "--", "-.", ":"])
    for method in METHODS:
        if method not in data:
            continue
        roc = data[method]
        ls = next(line_styles)
        pyplot.plot(roc["fpr"], roc["tpr"], linestyle=ls, label=METHODS[method])
    pyplot.axis((0, 1, 0, 1))
    pyplot.xlabel("fpr")
    pyplot.ylabel("tpr")
    pyplot.legend()
    pyplot.show()


def main(project: str, criterion: Criterion):
    results_path = "results"
    data = {}
    for file in sorted(os.listdir(results_path)):
        if project in file and criterion.value in file and file.endswith(".json"):
            match = re.match(rf"{project}-(.+)-{criterion.value}", file)
            method = match.group(1)
            with open(os.path.join(results_path, file)) as f:
                data[method] = json.load(f)
    plot_roc(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot ROC curves for a project')
    parser.add_argument('-p', '--project', type=str, help='project name', required=True)
    parser.add_argument("-C", "--criterion", choices=[e.value for e in Criterion], default="nodes",
                        help="the criterion on which the evaluation is done")
    args = parser.parse_args()
    main(args.project, Criterion(args.criterion))
