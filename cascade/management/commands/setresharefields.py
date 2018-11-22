# -*- coding: utf-8 -*-
import traceback

from bulk_update.helper import bulk_update
from django.core.management.base import BaseCommand
import time

from django.db.models import Q

from crud.models import Reshare


class Command(BaseCommand):
    help = 'Set null fields of resahres: user, ref_user, ref_datetime'

    def add_arguments(self, parser):
        pass

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()
            query_set = Reshare.objects.filter(
                Q(user__isnull=True) |
                Q(ref_user__isnull=True) |
                Q(ref_datetime__isnull=True))
            # query_set = query_set.values('id', 'post__author_id', 'reshared_post__author_id',
            #                                          'reshared_post__datetime')
            all_count = query_set.count()
            count = 0

            for resh in query_set.iterator():
                resh.user = resh.post.author
                resh.ref_user = resh.reshared_post.author
                resh.ref_datetime = resh.reshared_post.datetime
                resh.save()

                count += 1

                # if count % 10000 == 0:
                #     self.stdout.write('saved in db')
                #     self.stdout.write('time: %.2f s' % (time.time() - start))

                # Reshare.objects.filter(id=resh['id']).update(user_id=resh['post__author_id'],
                #                                              ref_user_id=resh['reshared_post__author_id'],
                #                                              ref_datetime=resh['reshared_post__datetime'])
                if count % 1000 == 0:
                    self.stdout.write('%d / %d reshares done' % (count, all_count))

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
