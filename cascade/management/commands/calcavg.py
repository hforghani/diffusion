# -*- coding: utf-8 -*-
import json
from optparse import make_option
import os
import traceback
from django.conf import settings
from django.core.management.base import BaseCommand
import time
from django.db.models import Q, Count
import numpy as np
from cascade.models import Project, ParamTypes
from crud.models import Reshare, UserAccount, Post
from scipy import sparse


class Command(BaseCommand):
    help = 'Calculates diffusion model parameters'

    option_list = BaseCommand.option_list + (
        make_option(
            "-p",
            "--project",
            type="string",
            dest="project",
            help="project name",
        ),
        make_option(
            "-w",
            "--weight",
            action="store_true",
            dest="weight",
            help="just calculate diffusion weights"
        ),
        make_option(
            "-d",
            "--delay",
            action="store_true",
            dest="delay",
            help="just calculate diffusion delays"
        ),
        make_option(
            "-c",
            "--continue",
            action="store_true",
            dest="continue",
            help="continue from the last point"
        ),
    )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()

            # Get project or raise exception.
            project_name = options['project']
            if project_name is None:
                raise Exception('project not specified')
            project = Project(project_name)
            train_set, test_set = project.load_data()

            self.stdout.write('querying posts and reshares ...')
            posts = Post.objects.filter(postmeme__meme_id__in=train_set).distinct().order_by('datetime')
            reshares = Reshare.objects.filter(post__in=posts, reshared_post__in=posts).distinct()

            if options['weight'] or not options['delay']:
                self.calc_weights(reshares, posts, project)
            if options['delay'] or not options['weight']:
                self.calc_delays(reshares, options['continue'], project)
            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise

    def calc_weights(self, reshares, posts, project):
        self.stdout.write('counting reshares of users ...')
        resh_counts = reshares.values('user_id', 'ref_user_id').annotate(count=Count('datetime'))

        self.stdout.write('counting posts of users ...')
        post_counts = list(posts.values('author_id').annotate(count=Count('datetime')))
        post_counts = {obj['author_id']: obj['count'] for obj in post_counts}

        self.stdout.write('getting user ids ...')
        user_ids = UserAccount.objects.values_list('id', flat=True)
        users_count = len(user_ids)
        users_map = {user_ids[i]: i for i in range(users_count)}

        self.stdout.write('constructing weight matrix ...')
        resh_count = resh_counts.count()
        values = np.zeros(resh_count)
        ij = np.zeros((2, resh_count))
        i = 0
        for resh in resh_counts:
            values[i] = float(resh['count']) / post_counts[resh['user_id']]
            sender_id = users_map[resh['ref_user_id']]
            receiver_id = users_map[resh['user_id']]
            ij[:, i] = [sender_id, receiver_id]
            i += 1
            if i % 10000 == 0:
                self.stdout.write('\t%d edges done' % i)
        del resh_counts
        weights = sparse.csc_matrix((values, ij), shape=(users_count, users_count))

        self.stdout.write('saving w ...')
        project.save_param(weights, 'w', ParamTypes.SPARSE)

    def calc_delays(self, reshares, continue_prev, project):
        self.stdout.write('prepairing user pairs ...')

        save_path = os.path.join('data', 'diffparam_delay_saved.npy')
        if continue_prev and os.path.exists(save_path):
            data = np.load(save_path).item()
            delays = data['delays']
            i = data['index']
            ignoring = True
            self.stdout.write('ignoring processed reshares ...')
        else:
            delays = {(pair['ref_user_id'], pair['user_id']): {'count': 0, 'avg': 0} for pair in
                      reshares.values('ref_user_id', 'user_id').distinct()}
            i = 0
            ignoring = False

        # Iterate on reshares. Average on the delays of reshares between each pair of users. Save it as diffusion delay
        # between them.
        j = 0
        for resh in reshares.iterator():
            if ignoring:
                if j < i:
                    j += 1
                    if j % (10 ** 6) == 0:
                        self.stdout.write('\t%d reshares ignored' % j)
                    continue
                else:
                    self.stdout.write('calculating diff. delays ...')
                    ignoring = False
            i += 1
            delay = (resh.datetime - resh.ref_datetime).total_seconds() / (3600.0 * 24)  # in days
            if delay > 0:
                delay_data = delays[(resh.ref_user_id, resh.user_id)]
                if not delay_data['count']:
                    delay_data['avg'] = delay
                else:
                    delay_data['avg'] = (delay_data['avg'] * delay_data['count'] + delay) / (delay_data['count'] + 1)
                delay_data['avg'] = np.array(delay_data['avg'], np.float16)
                delay_data['count'] += 1
            if i % 100000 == 0:
                self.stdout.write('\t%d reshares done' % i)

            # Save to continue in the future.
            if i % (10 ** 6) == 0:
                self.stdout.write('saving data ...')
                np.save(save_path, {'index': i, 'delays': delays})

        self.stdout.write('saving diff. delays in db ...')
        j = 0
        for pair in delays:
            j += 1
            if delays[pair]:
                delay = delays[pair]['avg']
                if np.isnan(delay):
                    delay = 1.0 / (24 * 60)  # 1 minute
                DiffusionParam.objects.filter(sender_id=pair[0], receiver_id=pair[1]).update(delay=delay)
            if j % 10000 == 0:
                self.stdout.write('\t%d diffusion delays set' % j)
