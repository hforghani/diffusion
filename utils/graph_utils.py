class Graph:
    """
    Less-memory implementation of networkx DiGraph
    """

    def __init__(self, graph):
        self.__predecessors = {node: list(graph.predecessors(node)) for node in graph}
        self.__successors = {node: list(graph.successors(node)) for node in graph}

    def predecessors(self, node):
        return self.__predecessors[node]

    def successors(self, node):
        return self.__successors[node]

    def in_degree(self, node):
        return len(self.__predecessors)

    def nodes(self):
        return list(self.__predecessors.keys())

    def edges(self):
        return ((pred, node) for node, pred_list in self.__predecessors.items() for pred in pred_list)

    def number_of_edges(self):
        return sum(len(pred_list) for pred_list in self.__predecessors.values())

    def __contains__(self, node):
        return node in self.__predecessors

    def __iter__(self):
        return iter(self.__predecessors)
