import argparse
import itertools
import json
import os
import re
from typing import Dict

from matplotlib import pyplot
import numpy as np

from diffusion.enum import Criterion

METHODS = {
    "aslt": "AsLT",
    "emic": "EMIC",
    "ctic": "CTIC",
    "daic": "DAIC",
    "infvae": "Inf-VAE",
    "deepdiffuse": "DeepDiffuse",
    "mbmemm": "SPMEMM",
    "mlmemm": "LPMEMM",
    "mbcrf": "SPCRF",
    "mlcrf": "LPCRF",
}


def plot_roc(data: Dict[str, Dict[str, np.array]]):
    """
    Save ROC plot as png and FPR-TPR values as json.
    """
    pyplot.figure()
    line_styles = itertools.cycle(["-", "--", "-.", ":"])
    line_colors = itertools.cycle("bgcmyk")
    for method in METHODS:
        if method not in data:
            continue
        roc = data[method]
        if method == "mlcrf":
            ls, line_width, color = "-", 4, "r"
        else:
            ls, line_width, color = next(line_styles), 1, next(line_colors)
        pyplot.plot(roc["fpr"], roc["tpr"],
                    label=METHODS[method],
                    linestyle=ls,
                    color=color,
                    linewidth=line_width)
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
