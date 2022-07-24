import abc
import logging
import multiprocessing
import os
import re

from bson import ObjectId
from scipy import sparse

import settings
from cascade.models import ParamTypes
from diffusion.enum import Method
from diffusion.models import DiffusionModel, IC
from log_levels import DEBUG_LEVELV_NUM
from mln.file_generators import TuffyCreator
from settings import logger


class MLN(abc.ABC):
    max_iterations = 100

    def learn_mln(self, train_set, project, multi_processed, iterations):
        creator = self.get_file_creator(project)
        proc_name = multiprocessing.current_process().name
        rules_file_name = f'/tmp/rules-{project.name}-{proc_name}.mln'
        train_evid_file_name = f'/tmp/evid-train-{project.name}-{proc_name}.db'
        creator.create_rules(train_set, rules_file_name)
        creator.create_train_evidence(train_set, train_evid_file_name)

        out_file_name = self.run_learn_script(rules_file_name, train_evid_file_name, project, multi_processed,
                                              iterations)

        # out_file_name = f'/tmp/out-{project.name}-{proc_name}.db'
        learn_results = self.parse_out_file(out_file_name)

        os.unlink(rules_file_name)
        os.unlink(train_evid_file_name)
        # os.unlink(out_file_name)

        return learn_results

    @abc.abstractmethod
    def get_file_creator(self, project):
        pass

    @abc.abstractmethod
    def run_learn_script(self, rules_file_name, train_evid_file_name, project, multi_processed, iterations):
        pass

    @abc.abstractmethod
    def parse_out_file(self, out_file_name):
        pass


class TuffyMLN(MLN, abc.ABC):
    def get_file_creator(self, project):
        return TuffyCreator(project)

    def run_learn_script(self, rules_file_name, train_evid_file_name, project, multi_processed, iterations):
        jar_file_name = os.path.join(settings.TUFFY_PATH, 'tuffy.jar')
        proc_name = multiprocessing.current_process().name
        query_file_name = f'/tmp/query-{project.name}-{proc_name}.db'
        with open(query_file_name, 'w') as f:
            f.write('activates(n1, n2, c)')
        out_file_name = f'/tmp/out-{project.name}-{proc_name}.db'
        verbosity = {
            logging.INFO: 0,
            logging.DEBUG: 1,
            DEBUG_LEVELV_NUM: 5
        }[settings.LOG_LEVEL]

        logger.info('running Java Tuffy command ...')
        max_iteration = iterations if iterations is not None else self.max_iterations
        threads = settings.TRAIN_WORKERS if multi_processed else 1
        os.system(
            f'java -jar {jar_file_name} -learnwt -i {rules_file_name} -e {train_evid_file_name} '
            f'-queryFile {query_file_name} -r {out_file_name} -mcsatSamples 50 -dMaxIter {max_iteration} '
            f' -threads {threads} -verbose {verbosity}')
        # os.system(
        #     f'java -jar {jar_file_name} -learnwt -mle -i {rules_file_name} -e {train_evid_file_name} '
        #     f'-queryFile {query_file_name} -r {out_file_name} -dMaxIter {max_iteration} -threads {threads} '
        #     f'-verbose {verbosity}')
        logger.info('Java Tuffy command done')

        os.unlink(query_file_name)

        return out_file_name
        # return out_file_name + '.prog'

    def parse_out_file(self, out_file_name):
        logger.info('parsing MLN results ...')
        regex = r'([\d\.-]+)\s+!isActivated\("\w+", v0\)  v  activates\("N(\w+)", "N(\w+)", v0\)'
        results = {}
        with open(out_file_name) as f:
            line = f.readline()
            while line:
                if 'WEIGHT OF LAST ITERATION' in line:
                    break
                match = re.match(regex, line)
                if match is not None:
                    groups = match.groups()
                    results[ObjectId(groups[1]), ObjectId(groups[2])] = float(groups[0])
                line = f.readline()
        return results


class TuffyMLNModel(DiffusionModel, TuffyMLN):
    # TODO: Directly predict via MLN results (either the weights of isActivated or activates rules).s
    pass


class TuffyICMLNModel(IC, TuffyMLN):
    method = Method.MLN_TUFFY
    max_iterations = 100

    def __init__(self, initial_depth=0, max_step=None, threshold=0.5, **kwargs):
        super().__init__(initial_depth, max_step, threshold)

    def calc_parameters(self, train_set, project, multi_processed, eco, iterations=None, **kwargs):
        learn_results = self.learn_mln(train_set, project, multi_processed, iterations)

        # Normalize the weights between 0 and 1.
        weights = list(learn_results.values())
        min_w, max_w = min(weights), max(weights)
        learn_results = {edge: (weight - min_w) / (max_w - min_w) for edge, weight in learn_results.items()}

        # Save the results in matrix k (probabilities).
        user_ids = sorted(self.graph.nodes())
        u_count = len(user_ids)
        user_map = {user_ids[i]: i for i in range(u_count)}
        self.k = sparse.lil_matrix((u_count, u_count))
        for edge, weight in learn_results.items():
            self.k[user_map[edge[0]], user_map[edge[1]]] = weight

        if eco:
            project.save_param(self.k, self.k_param_name, ParamTypes.SPARSE)

        self.k = self.k.tocsr()
