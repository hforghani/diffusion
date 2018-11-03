import os


class FileCreator:
    FORMAT_PRACMLN = 'pracmln'
    FORMAT_ALCHEMY2 = 'alchemy2'

    def __init__(self, project):
        self.project = project

        # Load training and test sets and all cascade trees.
        self.train_memes, self.test_memes = project.load_train_test()
        self.trees = project.load_trees()

    def create_rules(self, out_file):
        # Get the path of rules file.
        if out_file is not None:
            file_name = out_file
        else:
            file_name = '%s-%s-tolearn.mln' % (self.project.project_name, self.format)
        out_path = os.path.join(self.project.project_path, file_name)

        contents = self._get_contents()
        with open(out_path, 'w') as f:
            f.write(contents)

    def create_evidence(self, out_file):

        contents = self._get_contents()
        with open(out_path, 'w') as f:
            f.write(contents)

    def _get_contents(self):
        raise NotImplementedError('__get_contents most be implemented in the child')


class PracmlnCreator(FileCreator):
    def __init__(self, project):
        super().__init__(project)
        self.format = FileCreator.FORMAT_PRACMLN

    def _get_contents(self):
        res = ''
        print('training set size = %d' % len(self.train_memes))
        print('>>> writing declarations ...')
        res += '// predicate declarations\n' \
               'activates(user,user,meme)\n' \
               'isActivated(user,meme)\n\n'

        print('>>> writing rules ...')
        edges = set()
        for meme_id in self.train_memes:
            edges.update(self.trees[meme_id].edges())

        res += '//formulas\n'
        for sender, receiver in edges:
            res += '0     isActivated(u%d, ?m) => activates(u%d, u%d, ?m)\n' % (sender, sender, receiver)
        return res


class Alchemy2Creator(FileCreator):
    def __init__(self, project):
        super().__init__(project)
        self.format = FileCreator.FORMAT_ALCHEMY2

    def _get_contents(self):
        edges = set()

        print('>>> writing rules ...')
        for meme_id in self.train_memes:
            edges.update(self.trees[meme_id].edges())

        res = ''
        for sender, receiver in edges:
            res += 'isActivated(U%d, m) => activates(U%d, U%d, m)\n' % (sender, sender, receiver)
        return res
