# -*- coding: utf-8 -*-
import traceback
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
            all_count = Reshare.objects.count()
            count = 0

            for resh in Reshare.objects.filter(
                    Q(user__isnull=True) |
                    Q(ref_user__isnull=True) |
                    Q(ref_datetime__isnull=True)).iterator():
                # for resh in Reshare.objects.iterator():
                resh.user = resh.post.author
                resh.ref_user = resh.reshared_post.author
                resh.ref_datetime = resh.reshared_post.datetime
                resh.save()
                count += 1
                if count % 1000 == 0:
                    self.stdout.write('%d / %d reshares done' % (count, all_count))

            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
