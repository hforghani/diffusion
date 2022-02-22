# -*- coding: utf-8 -*-
import os
import re

from cascade.models import CascadeNode
from mln.file_generators import FileCreator
import settings
from settings import logger


class NodeProb:
    def __init__(self, node, prob):
        self.node = node
        self.prob = prob


class EdgeProb:
    def __init__(self, user1_id, user2_id, prob):
        self.user1_id = user1_id
        self.user2_id = user2_id
        self.prob = prob


class MLN(object):
    def __init__(self, project, format=FileCreator.FORMAT_PRACMLN, method='edge'):
        self.project = project
        self.edges = {}
        self.nodes = {}
        self.format = format
        self.method = method

    def load_edge_results(self, cascade_id, log=False):
        """
        Read "activates" predicates from MLN results file or get it from memory if exists.
        :param cascade_id: cascade id
        :param log:     whether log messages or not
        :return:        list of EdgeProb instances
        """
        if cascade_id in self.edges:
            return self.edges[cascade_id]

        else:
            self.edges[cascade_id] = []

            data_path = os.path.join(settings.BASE_PATH, 'data', self.project.name)
            if self.format == FileCreator.FORMAT_PRACMLN:
                results_file_path = os.path.join(data_path, 'results-pracmln',
                                                 '%s-m%d-gibbs.results' % (self.project.name, cascade_id))
            elif self.format == FileCreator.FORMAT_ALCHEMY2:
                if self.method == 'edge':
                    results_file_path = os.path.join(data_path, 'results-alchemy2-activates',
                                                     'results-%s-%s-m%d.results' % (
                                                         self.project.name, FileCreator.FORMAT_ALCHEMY2,
                                                         cascade_id))
                else:
                    results_file_path = os.path.join(data_path, 'results-alchemy2',
                                                     'results-%s-%s-m%d.results' % (
                                                         self.project.name, FileCreator.FORMAT_ALCHEMY2,
                                                         cascade_id))

            else:
                raise ValueError('invalid format "%s"' % self.format)

            if log:
                logger.info('loading mln results ...')

            if not os.path.exists(results_file_path):
                logger.warn('results file for cascade %d does not exist', cascade_id)
                return []

            with open(results_file_path) as f:
                line = f.readline()
                while line:
                    if self.format == FileCreator.FORMAT_PRACMLN:
                        regex = r'.+\s+([\d\.]+) % activates\(u(\d+),u(\d+),m(\d+)\)'
                    elif self.format == FileCreator.FORMAT_ALCHEMY2:
                        regex = r'activates\(U(\d+),U(\d+),M(\d+)\) (\S+)'
                    else:
                        raise ValueError('invalid format "%s"' % self.format)

                    match = re.search(regex, line)
                    if match is None:
                        continue
                    groups = match.groups()

                    if self.format == FileCreator.FORMAT_PRACMLN:
                        prob, user1, user2, cascade = float(groups[0]), int(groups[1]), int(groups[2]), int(groups[3])
                    else:
                        user1, user2, cascade, prob = int(groups[0]), int(groups[1]), int(groups[2]), float(groups[3])

                    self.edges[cascade].append(EdgeProb(user1, user2, prob))
                    line = f.readline()

            return self.edges[cascade_id]

    def load_node_results(self, cascade_id, log=False):
        """
        Read "isActivated" predicates from MLN results file or get it from memory if exists.
        :param cascade_id: cascade id
        :param log:     whether log messages or not
        :return:        list of NodeProb instances
        """
        if cascade_id in self.nodes:
            return self.nodes[cascade_id]

        else:
            self.nodes[cascade_id] = []

            data_path = os.path.join(settings.BASE_PATH, 'data', self.project.name)
            if self.format == FileCreator.FORMAT_PRACMLN:
                results_file_path = os.path.join(data_path, 'results-pracmln',
                                                 '%s-m%d-isActivated-gibbs.results' % (
                                                     self.project.name, cascade_id))
            elif self.format == FileCreator.FORMAT_ALCHEMY2:
                results_file_path = os.path.join(data_path, 'results-alchemy2',
                                                 'results-%s-%s-m%d.results' % (
                                                     self.project.name, FileCreator.FORMAT_ALCHEMY2, cascade_id))
            else:
                raise ValueError('invalid format "%s"' % self.format)

            if not os.path.exists(results_file_path):
                if log:
                    logger.warn('results file for cascade %d does not exist', cascade_id)
                return []

            if log:
                logger.info('loading mln results of cascade %d ...' % cascade_id)

            with open(results_file_path) as f:
                line = f.readline()
                while line:
                    if self.format == FileCreator.FORMAT_PRACMLN:
                        regex = r'.+\s+([\d\.]+) % isActivated\(u(\d+),m(\d+)\)'
                    elif self.format == FileCreator.FORMAT_ALCHEMY2:
                        regex = r'isActivated\(U(\d+),M(\d+)\) (\S+)'

                    match = re.search(regex, line)
                    if match is None:
                        continue
                    groups = match.groups()

                    if self.format == FileCreator.FORMAT_PRACMLN:
                        prob, user, cascade = float(groups[0]), int(groups[1]), int(groups[2])
                    elif self.format == FileCreator.FORMAT_ALCHEMY2:
                        user, cascade, prob = int(groups[0]), int(groups[1]), float(groups[2])

                    node = CascadeNode(user_id=user)
                    self.nodes[cascade].append(NodeProb(node, prob))
                    line = f.readline()

            return self.nodes[cascade_id]

    def predict(self, *args, **kwargs):
        if self.method == 'node':
            return self.predict_by_nodes(*args, **kwargs)
        elif self.method == 'edge':
            return self.predict_by_edges(*args, **kwargs)
        else:
            raise ValueError('invalid method "%s"' % self.method)

    def predict_by_edges(self, cascade_id, initial_tree, threshold, log=False):
        # TODO: Implement by thresholds instead of single threshold.
        # Load results of mln inference and put it in self.edges .
        edges = self.load_edge_results(cascade_id)

        res_tree = initial_tree.copy()
        if edges:
            predicted_edges = [edge_prob for edge_prob in edges if edge_prob.prob > threshold]
            self.__add_edges(res_tree, predicted_edges)
        else:
            if log:
                logger.warn('cascade id {} does not exist in MLN results'.format(cascade_id))
        return res_tree

    def predict_by_nodes(self, cascade_id, initial_tree, threshold, log=False):
        # TODO: Implement by thresholds instead of single threshold.
        # Load results of mln inference and put it in self.edges .
        nodes = self.load_node_results(cascade_id, log)

        res_tree = initial_tree.copy()
        any_node = res_tree.nodes()[0]

        if nodes:
            for node_prob in nodes:
                if node_prob.prob > threshold:
                    any_node.children.append(node_prob.node)
                    node_prob.node.parent_id = any_node.user_id
        else:
            if log:
                logger.warn('cascade id {} does not exist in MLN results'.format(cascade_id))
        return res_tree

    def __add_edges(self, tree, edge_probs):
        nodes = {node.user_id: node for node in tree.nodes()}
        child_added = True

        while child_added:
            child_added = False
            for edge in edge_probs:
                u1, u2 = edge.user1_id, edge.user2_id
                if u1 in nodes and u2 not in nodes:
                    node1 = nodes[u1]
                    node2 = CascadeNode(user_id=u2, parent_id=u1)
                    node1.children.append(node2)
                    nodes[node2.user_id] = node2
                    child_added = True
