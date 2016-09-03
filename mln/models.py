# -*- coding: utf-8 -*-
import re
from cascade.models import CascadeNode


class MLN(object):
    def __init__(self, project, threshold=50):
        self.project = project
        self.threshold = threshold

    def load_results(self, file_path, trees):
        edges = {}

        with open(file_path) as f:
            line = f.readline()
            while line:
                match = re.search(r'.+\s+([\d\.]+) % activates\(u(\d+),u(\d+),m(\d+)\)', line)
                if match is None:
                    continue
                groups = match.groups()
                percent, user1, user2, meme = float(groups[0]), int(groups[1]), int(groups[2]), int(groups[3])
                if percent >= self.threshold:
                    if meme not in edges:
                        edges[meme] = []
                    edges[meme].append({'user1': user1, 'user2': user2, 'p': percent})
                line = f.readline()

        predicted_trees = {}
        for meme_id in edges:
            res_tree = trees[meme_id].copy()
            self.add_edges(res_tree, edges[meme_id])
            predicted_trees[meme_id] = res_tree

        return predicted_trees

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
