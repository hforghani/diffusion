# -*- coding: utf-8 -*-
import os
import socket
import traceback
import urllib
from django.conf import settings
from django.core.management.base import BaseCommand
import time
from cascade.models import CascadeTree
from crud.models import UserAccount, SocialNet, Meme, Reshare


class Command(BaseCommand):
    help = 'Calculate meme depths.'

    def handle(self, *args, **options):
        try:
            start = time.time()
            self.stdout.write('meme count = %d' % Meme.objects.count())
            i = 0
            for meme in Meme.objects.filter(depth__isnuall=True).iterator():
                tree = CascadeTree().extract_cascade(meme)
                meme.depth = tree.depth
                meme.save()
                i += 1
                if i % 10 == 0:
                    self.stdout.write('%d memes done' % i)

            self.stdout.write('command done in %.2f min' % ((time.time() - start) / 60.0))
        except:
            self.stdout.write(traceback.format_exc())
            raise
