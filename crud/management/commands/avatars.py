# -*- coding: utf-8 -*-
import os
import socket
import traceback
import urllib
from django.conf import settings
from django.core.management.base import BaseCommand
import time
from crud.models import UserAccount, SocialNet


class Command(BaseCommand):
    help = 'Download avatar images into media'

    def handle(self, *args, **options):
        try:
            start = time.time()
            socket.setdefaulttimeout(5)
            i = 0

            for net in SocialNet.objects.all():
                save_dir = os.path.join(settings.BASEPATH, 'media', 'avatars', net.name)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                for user in UserAccount.objects.filter(social_net=net).iterator():
                    i += 1
                    if user.avatar:
                        fname = '%s%s' % (user.username, user.avatar[-4:])
                        path = os.path.join(save_dir, fname)
                        if not os.path.exists(path):
                            try:
                                res = urllib.urlretrieve(user.avatar, path)
                                if res[1].maintype == 'image':
                                    user.avatar = '/media/avatars/%s/%s' % (user.social_net.name, fname)
                                    self.stdout.write('%d: saved' % i)
                                else:
                                    os.remove(path)
                                    user.avatar = None
                                    self.stdout.write('%d: unable to download' % i)
                            except IOError:
                                user.avatar = None
                                self.stdout.write('%d: unable to download' % i)
                            user.save()
                            continue
                    self.stdout.write('%d: ignored' % i)
            self.stdout.write('command done in %.2f min' % ((time.time() - start) / 60.0))
        except:
            self.stdout.write(traceback.format_exc())
            raise
