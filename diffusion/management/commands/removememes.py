# -*- coding: utf-8 -*-
import traceback
from django.core.management.base import BaseCommand
import time
from django.db.models import Q
from crud.models import Meme


class Command(BaseCommand):
    help = ''

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
        try:
            start = time.time()
            memes = Meme.objects.filter(Q(count__isnull=True) | Q(count__lt=10))
            meme_ids = memes.values_list('id', flat=True)
            count = memes.count()
            self.stdout.write('deleting %d memes ...' % count)
            step = 1000
            for i in range(step, count + step, step):
                memes.filter(id__in=meme_ids[i-step: i]).delete()
                self.stdout.write('%d memes deleted' % min(i, count))
            memes.delete()
            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
