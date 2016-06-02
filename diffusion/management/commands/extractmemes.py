# -*- coding: utf-8 -*-
import traceback
from django.core.management.base import BaseCommand, CommandError
import time
from crud.models import Post, Meme, PostMeme
from diffusion.models import MemeDetector


class Command(BaseCommand):
    help = 'Extracts memes from posts'

    def handle(self, *args, **options):
        try:
            start = time.time()
            MemeDetector().extract_memes()
            self.stdout.write('command done in %f min' % ((time.time() - start) / 60))
        except:
            self.stdout.write(traceback.format_exc())
            raise
