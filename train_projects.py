import argparse

from cascade.models import Project
from cascade.testers import MultiProcTester, DefaultTester
from diffusion.enum import Method


def main(args):
    for i in range(args.min_project_num, args.max_project_num + 1):
        project = Project(f'{args.db}-analysis-{i}')

        for meth_name in args.methods:
            method = Method(meth_name)
            if args.multi_processed:
                tester = MultiProcTester(project, method)
            else:
                tester = DefaultTester(project, method)
            tester.train(iterations=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a project with the method given')
    parser.add_argument("-n", "--min-num", type=int, dest="min_project_num", default=1, help="min project name")
    parser.add_argument("-N", "--max-num", type=int, dest="max_project_num", required=True, help="max project name")
    parser.add_argument("-d", "--db", type=str, required=True, help="db name")
    parser.add_argument("-m", "--methods", nargs='+', required=True, choices=[e.value for e in Method],
                        help="the methods by which we want to test")
    parser.add_argument("-M", "--multiprocessed", action='store_true', dest="multi_processed", default=False,
                        help="If this option is given, the task is ran on multiple processes")
    args = parser.parse_args()

    main(args)
