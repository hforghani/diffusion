# -*- coding: utf-8 -*-
import logging
import os
import re

from django.conf import settings

from cascade.models import CascadeNode
from mln.file_generators import FileCreator

logger = logging.getLogger('mln.models')


class MLN(object):
    def __init__(self, project, format=FileCreator.FORMAT_PRACMLN):
        self.project = project
        self.edges = {}
        self.format = format

    def load_results(self, meme_id):
        if meme_id in self.edges:
            return self.edges[meme_id]
        else:
            self.edges[meme_id] = []

        data_path = os.path.join(settings.BASEPATH, 'data', self.project.project_name)
        if self.format == FileCreator.FORMAT_PRACMLN:
            results_file_path = os.path.join(data_path, 'results-pracmln',
                                             '%s-m%d-gibbs.results' % (self.project.project_name, meme_id))
        elif self.format == FileCreator.FORMAT_ALCHEMY2:
            results_file_path = os.path.join(data_path, 'results-alchemy2',
                                             'results-%s-%s-m%d.results' % (
                                                 self.project.project_name, FileCreator.FORMAT_ALCHEMY2, meme_id))
        else:
            raise ValueError('invalid format "%s"' % self.format)

        logger.info('loading mln results ...')

        if not os.path.exists(results_file_path):
            logger.warning('results file for meme %d does not exist', meme_id)
            return

        with open(results_file_path) as f:
            line = f.readline()
            while line:
                if self.format == FileCreator.FORMAT_PRACMLN:
                    regex = r'.+\s+([\d\.]+) % activates\(u(\d+),u(\d+),m(\d+)\)'
                else:
                    regex = r'activates\(U(\d+),U(\d+),M(\d+)\) (\S+)'

                match = re.search(regex, line)
                if match is None:
                    continue
                groups = match.groups()

                if self.format == FileCreator.FORMAT_PRACMLN:
                    percent, user1, user2, meme = float(groups[0]), int(groups[1]), int(groups[2]), int(groups[3])
                else:
                    user1, user2, meme, percent = int(groups[0]), int(groups[1]), int(groups[2]), float(groups[3])

                self.edges[meme].append({'user1': user1, 'user2': user2, 'p': percent})
                line = f.readline()

    def predict(self, meme_id, initial_tree, threshold=30):
        # Load results of mln inference and put it in self.edges .
        self.load_results(meme_id)

        res_tree = initial_tree.copy()
        if meme_id in self.edges:
            predicted_edges = [edge for edge in self.edges[meme_id] if edge['p'] > threshold]
            self.add_edges(res_tree, predicted_edges)
        else:
            print('WARNING: meme id {} does not exists in MLN results'.format(meme_id))
        return res_tree

    def add_edges(self, tree, edges):
        nodes = {node.user_id: node for node in tree.nodes()}
        child_added = True

        while child_added:
            child_added = False
            for edge in edges:
                u1, u2 = edge['user1'], edge['user2']
                if u1 in nodes and u2 not in nodes:
                    node1 = nodes[u1]
                    node2 = CascadeNode(user_id=edge['user2'], parent_id=edge['user1'])
                    node1.children.append(node2)
                    nodes[node2.user_id] = node2
                    child_added = True
