import logging
import os
import numpy
from django.db.models import Count
from scipy import sparse

from crud.models import UserAccount, Post, Reshare

logger = logging.getLogger('mln.file_generators')


class FileCreator:
    FORMAT_PRACMLN = 'pracmln'
    FORMAT_ALCHEMY2 = 'alchemy2'

    def __init__(self, project):
        self.project = project

        # Load training and test sets and all cascade trees.
        self.train_memes, self.test_memes = project.load_train_test()
        self.trees = project.load_trees()
        self.__follow_thr = 0.1

        # These fields must be set in children:
        self.format = None
        self.put_declarations = None
        self.put_zero_weights = None
        self.user_prefix = None
        self.meme_prefix = None
        self.meme_var_name = None

    def create_rules(self, out_file):
        contents = ''
        print('training set size = %d' % len(self.train_memes))
        if self.put_declarations:
            print('>>> writing declarations ...')
            contents += '// predicate declarations\n' \
                        'activates(user,user,meme)\n' \
                        'isActivated(user,meme)\n\n'

        print('>>> writing rules ...')
        edges = set()
        for meme_id in self.train_memes:
            edges.update(self.trees[meme_id].edges())

        contents += '//formulas\n'
        for sender, receiver in edges:
            if self.put_zero_weights:
                contents += '0     '
            contents += 'isActivated({0}{1}, {3}) => activates({0}{1}, {0}{2}, {3})\n'.format(self.user_prefix, sender,
                                                                                              receiver,
                                                                                              self.meme_var_name)

        # Get the path of rules file.
        if out_file is not None:
            file_name = out_file
        else:
            file_name = 'tolearn-%s-%s.mln' % (self.project.project_name, self.format)
        out_path = os.path.join(self.project.project_path, file_name)

        with open(out_path, 'w') as f:
            f.write(contents)

    def create_evidence(self, target_set, multiple):
        if target_set is None or target_set == 'train':
            # Get and delete the content of evidence file.
            out_file = os.path.join(self.project.project_path,
                                    'ev-train-%s-%s.db' % (self.project.project_name, self.format))
            open(out_file, 'w')

            logger.info('rules will be created for training set')
            logger.info('training set size = %d' % len(self.train_memes))

            logger.info('>>> writing "isActivated" rules ...')
            self.__write_isactivated(self.trees, self.train_memes, out_file)

            logger.info('>>> writing "activates" rules ...')
            self.__write_activates(self.trees, self.train_memes, out_file)

            # logger.info('>>> writing "follows" rules ...')
            # user_ids = UserAccount.objects.values_list('id', flat=True)
            # user_indexes = {user_ids[i]: i for i in range(len(user_ids))}
            # self.__write_follows(user_ids, user_indexes, self.train_memes, out_file)

        if target_set is None or target_set == 'test':
            # Get and delete the content of evidence file.
            out_file = os.path.join(self.project.project_path,
                                    'ev-test-%s-%s.db' % (self.project.project_name, self.format))
            open(out_file, 'w')

            logger.info('rules will be created for test set')
            logger.info('test set size = %d' % len(self.test_memes))
            logger.info('>>> writing "isActivated" rules ...')
            self.__write_isactivated(self.trees, self.test_memes, out_file, initials=True)

            if multiple:
                logger.info('>>> writing "isActivated" rules ...')
                for meme_id in self.test_memes:
                    meme_out_file = os.path.join(self.project.project_path,
                                                 'ev-test-%s-%s-m%d.db' % (
                                                 self.project.project_name, self.format, meme_id))
                    self.__write_isactivated(self.trees, [meme_id], meme_out_file, initials=True)

    def __write_follows(self, user_ids, user_indexes, train_memes, out_file):
        post_ids = Post.objects.filter(postmeme__meme_id__in=train_memes).values_list('id', flat=True).distinct()
        reshares = Reshare.objects.filter(post__in=post_ids, reshared_post__in=post_ids)

        logger.info('counting total reshares ...')
        users_count = len(user_ids)
        total_resh_count = numpy.zeros(users_count)
        for resh in reshares.values('user_id').annotate(count=Count('id')):
            total_resh_count[user_indexes[resh['user_id']]] = resh['count']

        logger.info('counting pairwise reshares ...')
        resh_count = sparse.lil_matrix((users_count, users_count))
        for resh in reshares.values('ref_user_id', 'user_id').annotate(count=Count('id')):
            u1 = user_indexes[resh['ref_user_id']]
            u2 = user_indexes[resh['user_id']]
            resh_count[u1, u2] = resh['count']

        logger.info('writing followship rules ...')
        with open(out_file, 'a') as f:
            rows, cols = resh_count.nonzero()
            for i in range(len(rows)):
                u1, u2 = rows[i], cols[i]
                if u1 == u2:
                    continue
                ratio = float(resh_count[u1, u2]) / total_resh_count[u2]
                if ratio > self.__follow_thr:
                    f.write('follows({0}{1}, {0}{2})\n'.format(self.user_prefix, user_ids[u2], user_ids[u1]))
            f.write('\n')

    def __write_isactivated(self, trees, meme_ids, out_file, initials=False):
        with open(out_file, 'a') as f:
            for meme_id in meme_ids:
                if initials:
                    nodes = trees[meme_id].roots
                else:
                    nodes = trees[meme_id].nodes()
                for node in nodes:
                    f.write('isActivated({0}{2}, {1}{3})\n'.format(self.user_prefix, self.meme_prefix, node.user_id,
                                                                   meme_id))
            f.write('\n')

    def __write_activates(self, trees, meme_ids, out_file):
        with open(out_file, 'a') as f:
            for meme_id in meme_ids:
                edges = trees[meme_id].edges()
                for edge in edges:
                    f.write('activates({0}{2}, {0}{3}, {1}{4})\n'.format(self.user_prefix, self.meme_prefix, edge[0],
                                                                         edge[1], meme_id))
            f.write('\n')


class PracmlnCreator(FileCreator):
    def __init__(self, project):
        super().__init__(project)
        self.format = FileCreator.FORMAT_PRACMLN
        self.put_declarations = True
        self.put_zero_weights = True
        self.user_prefix = 'u'
        self.meme_prefix = 'm'
        self.meme_var_name = '?m'


class Alchemy2Creator(FileCreator):
    def __init__(self, project):
        super().__init__(project)
        self.format = FileCreator.FORMAT_ALCHEMY2
        self.put_declarations = True
        self.put_zero_weights = False
        self.user_prefix = 'U'
        self.meme_prefix = 'M'
        self.meme_var_name = 'm'
