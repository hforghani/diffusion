# -*- coding: utf-8 -*-
from optparse import make_option
import traceback
from django.db.models import Count
import numpy as np
from django.core.management.base import BaseCommand
import time
from scipy import sparse
from crud.models import Meme, UserAccount, Reshare


class Command(BaseCommand):
    help = ''

    option_list = BaseCommand.option_list + (
        make_option(
            "-n",
            "--number",
            type="int",
            dest="train_num",
            help="number of training samples"
        ),
    )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()

            train_num = options['train_num']
            train_num = min(Meme.objects.count(), train_num)

            users_count = UserAccount.objects.count()
            user_ids = UserAccount.objects.values_list('id', flat=True)
            user_indexes = {user_ids[i]: i for i in range(len(user_ids))}
            total_resh = np.zeros(users_count)

            self.stdout.write('counting total reshares ...')
            for resh in Reshare.objects.values('user_id').annotate(count=Count('id')):
                total_resh[user_indexes[resh['user_id']]] = resh['count']

            self.stdout.write('counting total reshares ...')
            reshares = sparse.lil_matrix((users_count, users_count))

            for resh in Reshare.objects.values('ref_user_id', 'user_id').annotate(count=Count('id')):
                u1 = user_indexes[resh['ref_user_id']]
                u2 = user_indexes[resh['user_id']]
                reshares[u1, u2] = resh['count']

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))

        except:
            self.stdout.write(traceback.format_exc())
            raise
