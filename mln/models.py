# -*- coding: utf-8 -*-
import re
from cascade.models import CascadeNode


class MLN(object):
    def __init__(self, project):
        self.project = project
        self.edges = {}
        self.results_file_path = 'D:/University Stuff/social/code/pracmln/experiments/social/results/%s-gibbs.results' % project.project_name

    def load_results(self):
        with open(self.results_file_path) as f:
            line = f.readline()
            while line:
                match = re.search(r'.+\s+([\d\.]+) % activates\(u(\d+),u(\d+),m(\d+)\)', line)
                if match is None:
                    continue
                groups = match.groups()
                percent, user1, user2, meme = float(groups[0]), int(groups[1]), int(groups[2]), int(groups[3])
                if meme not in self.edges:
                    self.edges[meme] = []
                self.edges[meme].append({'user1': user1, 'user2': user2, 'p': percent})
                line = f.readline()

    def predict(self, meme_id, initial_tree, threshold=30):
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
