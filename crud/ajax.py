# -*- coding: utf-8 -*-
from copy import copy
import logging
from django.contrib.auth.decorators import login_required
from django.db.models import DateTimeField
from django.http.response import HttpResponse
from django.db.models.fields.related import ManyToManyField
from crud.models import *

logger = logging.getLogger('diffusion.crud.ajax')


@login_required
def main(request):
    res = {}
    if request.method == 'GET':
        table = request.GET['collection']
        action = request.GET['action']
        model_class = get_model_class(table)
        if model_class:
            spec = dict(copy(request.GET))
            del spec['collection']
            del spec['action']
            for key, value in spec.items():
                if isinstance(value, list):
                    spec[key] = value[0]

            if action == 'read':
                res = handle_read(model_class, spec)
            elif action == 'create':
                res = handle_create(model_class, spec)
            elif action == 'delete':
                handle_delete(model_class, spec)
            elif action == 'update':
                handle_update(model_class, spec)
            else:
                raise ValueError('invalid action')
        else:
            raise ValueError('invalid collection name')

    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')


def dic_matches(dic, spec):
    for key in spec:
        if dic[key] != spec[key]:
            return False
    return True


def get_model_class(table_name):
    if table_name == 'user_account':
        return UserAccount
    elif table_name == 'social_net':
        return SocialNet
    elif table_name == 'post':
        return Post
    elif table_name == 'reshare':
        return Reshare
    elif table_name == 'friendship':
        return Friendship
    elif table_name == 'meme':
        return Meme
    elif table_name == 'post_meme':
        return PostMeme
    else:
        return None


def handle_read(model_class, spec):
    from_index = None
    to_index = None
    if 'from' in spec:
        from_index = int(spec['from'])
        del spec['from']
    if 'to' in spec:
        to_index = int(spec['to'])
        del spec['to']
    res = model_class.objects.filter(**spec)[from_index: to_index]
    res = [to_dict(instance) for instance in res]
    return res


def handle_create(model_class, spec):
    oid = model_class.objects.create(**spec).id
    res = {'id': oid}
    return res


def handle_delete(model_class, spec):
    model_class.objects.filter(**spec).delete()


def handle_update(model_class, spec):
    oid = spec['id']
    del spec['id']
    model_class.objects.filter(id=oid).update(**spec)


def to_dict(instance):
    opts = instance._meta
    data = {}
    for f in opts.fields + opts.many_to_many:
        if isinstance(f, ManyToManyField):
            if instance.pk is None:
                data[f.name] = []
            else:
                data[f.name] = list(f.value_from_object(instance).values_list('pk', flat=True))
        else:
            data[f.name] = f.value_from_object(instance)
        if isinstance(f, DateTimeField):
            data[f.name] = str(data[f.name])
    return data
