# -*- coding: utf-8 -*-
import json
from optparse import make_option
import os
import traceback
from django.conf import settings
from django.db.models import Count
import numpy as np
from django.core.management.base import BaseCommand
import time
from scipy import sparse
from cascade.models import CascadeTree, Project
from crud.models import Meme, UserAccount, Reshare, Post


class Command(BaseCommand):
    help = ''

    option_list = BaseCommand.option_list + (
        make_option(
            "-p",
            "--project",
            type="string",
            dest="project",
            help="project name",
        ),
        make_option(
            "-o", "--output",
            type="string",
            dest="out_file",
            help="path of output file",
        ),
    )

    def handle(self, *args, **options):
        try:
            start = time.time()

            # Get project or raise exception.
            project_name = options['project']
            if project_name is None:
                raise Exception('project not specified')
            project = Project(project_name)

            # Load training and test sets and all cascade trees.
            train_memes, test_memes = project.load_data()
            trees = project.load_trees()

            # Get the path of rules file.
            if options['out_file']:
                file_name = options['out_file']
            else:
                file_name = 'rules.mln'
            out_file = os.path.join(project.project_path, file_name)

            self.stdout.write('training set size = %d' % len(train_memes))
            self.stdout.write('>>> writing declarations ...')
            with open(out_file, 'w') as f:
                f.write('// predicate declarations\n'
                        'activates(user,user,meme)\n'
                        'isActivated(user,meme)\n\n')

            self.stdout.write('>>> writing rules ...')
            self.write_formulas(trees, train_memes, out_file)

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))

        except:
            self.stdout.write(traceback.format_exc())
            raise

    def write_formulas(self, trees, meme_ids, out_file):
        edges = set()

        for meme_id in meme_ids:
            edges.update(trees[meme_id].edges())

        with open(out_file, 'a') as f:
            f.write('//formulas\n')
            for sender, receiver in edges:
                f.write('0     isActivated(u%d, ?m) => activates(u%d, u%d, ?m)\n' % (sender, sender, receiver))
