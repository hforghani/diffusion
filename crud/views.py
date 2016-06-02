# -*- coding: utf-8 -*-
import os
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render_to_response
from django.template import RequestContext
from crud.models import DataEntry, SocialNet, UserAccount, Person, Post, Comment, Friendship, Like, Reshare


@login_required
def entry_list(request):
    entries = []
    if os.path.exists(settings.DATA_ENTRY_PATH):
        for name in os.listdir(settings.DATA_ENTRY_PATH):
            if os.path.isdir(os.path.join(settings.DATA_ENTRY_PATH, name)):
                entries.append(name)
    return render_to_response('crud/entry_list.html', {'entries': entries}, RequestContext(request))


@login_required
def entry(request):
    if request.method == 'GET':
        name = request.GET['name']
        try:
            entry = DataEntry.objects.get(name=name)
        except ObjectDoesNotExist:
            entry = DataEntry.objects.create(name=name)
        return render_to_response('crud/entry.html', {'entry': entry}, RequestContext(request))
    else:
        raise Exception('No data entry name given')
