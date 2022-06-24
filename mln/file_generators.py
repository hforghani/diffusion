import os
from settings import logger


class FileCreator:
    FORMAT_PRACMLN = 'pracmln'
    FORMAT_ALCHEMY2 = 'alchemy2'
    FORMAT_TUFFY = 'tuffy'

    def __init__(self, project):
        self.project = project

        self.trees = project.load_trees()
        self.__follow_thr = 0.1

        # These fields must be set in children:
        self.declarations = []
        self.format = None
        self.initial_weight = None
        self.hard_formulas = []
        self.node_prefix = None
        self.cascade_prefix = None
        self.node_var_name = None
        self.cascade_var_name = None

    def create_rules(self, train_set, out_file):
        contents = ''
        logger.info('started to write MLN rules')
        logger.debug('training set size = %d' % len(train_set))
        logger.debug('>>> writing declarations ...')
        if self.declarations:
            contents += '\n'.join(['// predicate declarations'] + self.declarations) + '\n'

        logger.debug('>>> writing rules ...')

        if self.hard_formulas:
            logger.debug('>>> writing hard formulas ...')
            contents += '\n'.join(self.hard_formulas) + '\n'

        # Write the learnable formulas.
        graph = self.project.load_or_extract_graph(train_set)
        contents += '//formulas\n'
        for sender, receiver in graph.edges():
            if self.initial_weight is not None:
                contents += f'{self.initial_weight}     '
            contents += 'isActivated({0}{1}, {3}) => activates({0}{1}, {0}{2}, {3})\n'.format(self.node_prefix,
                                                                                              sender, receiver,
                                                                                              self.cascade_var_name)

        # Get the path of rules file.
        # file_name = 'tolearn-%s-%s.mln' % (self.project.name, self.format)
        out_path = os.path.join(self.project.path, out_file)
        with open(out_path, 'w') as f:
            f.write(contents)

    def create_train_evidence(self, train_set, train_out_file):
        logger.debug('loading graph ...')
        graph = self.project.load_or_extract_graph(train_set)
        predecessors = self.__pred_predicates(graph)

        # Create the training evidence file.
        logger.info('started to write evidences for learning')
        logger.debug('training set size = %d' % len(train_set))
        # train_file = os.path.join(out_dir, 'ev-train-%s-%s.db' % (self.project.name, self.format))
        with open(train_out_file, 'w') as f:
            logger.debug('>>> writing "predecessor" rules ...')
            f.write('\n'.join(predecessors) + '\n\n')
            logger.debug('>>> writing "isActivated" rules ...')
            f.write('\n'.join(self.__isactivated_predicates(self.trees, train_set)) + '\n\n')
            logger.debug('>>> writing "activates" rules ...')
            f.write('\n'.join(self.__activates_predicates(self.trees, train_set)) + '\n\n')

    def create_test_evidence(self, train_set, test_set, test_out_file, multiple=False):
        logger.debug('loading graph ...')
        graph = self.project.load_or_extract_graph(train_set)
        predecessors = self.__pred_predicates(graph)

        # Create the test evidence file.
        logger.info('started to write evidences for inference')
        logger.debug('test set size = %d' % len(test_set))
        with open(test_out_file, 'w') as f:
            logger.debug('>>> writing "predecessor" rules ...')
            f.write('\n'.join(predecessors) + '\n\n')
            logger.debug('>>> writing "isActivated" rules ...')
            f.write('\n'.join(self.__isactivated_predicates(self.trees, test_set, initials=True)) + '\n\n')

        # if multiple:
        #     logger.debug('>>> writing "isActivated" rules ...')
        #     for cascade_id in cascades:
        #         cascade_out_file = os.path.join(self.project.path, f'evidence-{self.format}',
        #                                         f'ev-test-{self.project.name}-{self.format}-{cascade_id}.db')
        #         open(out_file, 'w').close()
        #         self.__isactivated_predicates(self.trees, [cascade_id], cascade_out_file, initials=True)

    def __pred_predicates(self, graph):
        predicates = ['predecessor({0}{1}, {0}{2})'.format(self.node_prefix, edge[0], edge[1]) for edge in
                      graph.edges()]
        return predicates

    def __isactivated_predicates(self, trees, cascade_ids, initials=False):
        predicates = []
        for cascade_id in cascade_ids:
            nodes = trees[cascade_id].roots if initials else trees[cascade_id].nodes()
            predicates.extend('isActivated({0}{2}, {1}{3})'.format(self.node_prefix, self.cascade_prefix, node.user_id,
                                                                   cascade_id) for node in nodes)
        return predicates

    def __activates_predicates(self, trees, cascade_ids):
        predicates = []
        for cid in cascade_ids:
            edges = trees[cid].edges()
            predicates.extend('activates({0}{2}, {0}{3}, {1}{4})'.format(self.node_prefix, self.cascade_prefix,
                                                                         edge[0], edge[1], cid) for edge in edges)
        return predicates


class PracmlnCreator(FileCreator):
    def __init__(self, project):
        super().__init__(project)
        self.format = FileCreator.FORMAT_PRACMLN
        self.declarations = [
            'activates(node,node,cascade)',
            'isActivated(node,cascade)',
            'predecessor(node, node)'
        ]

        self.initial_weight = 0
        self.node_prefix = 'n'
        self.cascade_prefix = 'c'
        self.node_var_name = '?n'
        self.cascade_var_name = '?c'


class Alchemy2Creator(FileCreator):
    def __init__(self, project):
        super().__init__(project)
        self.format = FileCreator.FORMAT_ALCHEMY2
        self.declarations = [
            'activates(node,node,cascade)',
            'isActivated(node,cascade)',
            'predecessor(node, node)'
        ]
        self.hard_formulas = [
            '!activates(n1, n1, c).',
            'activates(n1, n2, c) => isActivated(n2, c).',
            '!(activates(n1, n3, c) ^ activates(n2, n3, c) ^ (n1 != n2)).'
        ]
        self.node_prefix = 'N'
        self.cascade_prefix = 'C'
        self.node_var_name = 'n'
        self.cascade_var_name = 'c'


class TuffyCreator(Alchemy2Creator):
    def __init__(self, project):
        super().__init__(project)
        self.format = FileCreator.FORMAT_TUFFY
        self.declarations = [
            'activates(node!, node, cascade)',
            'isActivated(node, cascade)',
            'predecessor(node, node)'
        ]
        self.hard_formulas = [
            '!activates(n1, n1, c).',
            'activates(n1, n2, c) => isActivated(n2, c).',
            'activates(n1, n2, c) := predecessor(n1, n2), isActivated(n1, c), isActivated(n2, c).'
        ]
        self.initial_weight = 1
