# -*- coding: utf-8 -*-
from copy import copy
import logging
import traceback
from django.core.exceptions import ObjectDoesNotExist
import unicodecsv as csv
import json
from threading import Thread
from django.contrib.auth.decorators import login_required
from django.db.models import DateTimeField
from django.http.response import HttpResponse
from django.db.models.fields.related import ManyToManyField
from crud.models import *
from utils.time_utils import str_to_datetime

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


@login_required
def person_education(request):
    res = {'ok': False}
    if request.method == 'GET':
        action = request.GET['action']
        person_id = request.GET['person']
        person = Person().get({'id': person_id})
        educations = person.get('educations', [])
        spec = dict(copy(request.GET))
        del spec['action']
        del spec['person']

        if action == 'read':
            res = []
            for edu in educations:
                if dic_matches(edu, spec):
                    res.append(edu)
        elif action == 'create':
            for edu in educations:
                if dic_matches(edu, spec):
                    raise ValueError('education already exists')
            educations.append(spec)
            Person().update({'id': person_id}, {'educations': educations})
            res = {'ok': True}
        elif action == 'delete':
            for edu in educations:
                if dic_matches(edu, spec):
                    educations.remove(edu)
                    Person().update({'id': person_id}, {'educations': educations})
                    res = {'ok': True}
                    break
            if not res['ok']:
                raise ValueError('education does not exist')
        else:
            raise ValueError('invalid action')

    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')


@login_required
def person_work(request):
    res = {'ok': False}
    if request.method == 'GET':
        action = request.GET['action']
        person_id = request.GET['person']
        person = Person().get({'id': person_id})
        works = person.get('works', [])
        spec = dict(copy(request.GET))
        del spec['action']
        del spec['person']

        if action == 'read':
            res = []
            for work in works:
                if dic_matches(work, spec):
                    res.append(work)
        elif action == 'create':
            for work in works:
                if dic_matches(work, spec):
                    raise ValueError('education already exists')
            works.append(spec)
            Person().update({'id': person_id}, {'works': works})
            res = {'ok': True}
        elif action == 'delete':
            for work in works:
                if dic_matches(work, spec):
                    works.remove(work)
                    Person().update({'id': person_id}, {'works': works})
                    res = {'ok': True}
                    break
            if not res['ok']:
                raise ValueError('education does not exist')
        else:
            raise ValueError('invalid action')

    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')


def dic_matches(dic, spec):
    for key in spec:
        if dic[key] != spec[key]:
            return False
    return True


def get_model_class(table_name):
    if table_name == 'person':
        return Person
    elif table_name == 'user_account':
        return UserAccount
    elif table_name == 'social_net':
        return SocialNet
    elif table_name == 'post':
        return Post
    elif table_name == 'like':
        return Like
    elif table_name == 'comment':
        return Comment
    elif table_name == 'reshare':
        return Reshare
    elif table_name == 'friendship':
        return Friendship
    elif table_name == 'group':
        return Group
    elif table_name == 'group_membership':
        return GroupMembership
    elif table_name == 'hashtag':
        return Hashtag
    elif table_name == 'hashtag_post':
        return HashtagPost
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


@login_required
def entry_stat(request):
    """
    input via GET:
        name = <entry_name>
    output:
        {
        started: <whether_the_process_has_started>,
        finished: <whether_the_process_has_finished>,
        nets: {progress: <percent>, error: <error_message_if_exists>},
        users: {progress: <percent>, error: <error_message_if_exists>},
        posts: {progress: <percent>, error: <error_message_if_exists>},
        comments: {progress: <percent>, error: <error_message_if_exists>},
        likes: {progress: <percent>, error: <error_message_if_exists>},
        reshares: {progress: <percent>, error: <error_message_if_exists>},
        friendships: {progress: <percent>, error: <error_message_if_exists>},
        }
    """
    res = {}
    if request.method == 'GET':
        name = request.GET['name']
        entry = DataEntry.objects.get(name=name)
        res = {
            'started': entry.started,
            'finished': entry.finished,
            'nets': {'progress': entry.nets_progress, 'error': entry.nets_message},
            'users': {'progress': entry.users_progress, 'error': entry.users_message},
            'posts': {'progress': entry.posts_progress, 'error': entry.posts_message},
            'comments': {'progress': entry.comments_progress, 'error': entry.comments_message},
            'likes': {'progress': entry.likes_progress, 'error': entry.likes_message},
            'reshares': {'progress': entry.reshares_progress, 'error': entry.reshares_message},
            'friendships': {'progress': entry.friends_progress, 'error': entry.friends_message},
        }

    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')


@login_required
def start_entry_process(request):
    res = {'ok': False}
    if request.method == 'GET':
        name = request.GET['name']
        entry = DataEntry.objects.get(name=name)
        path = os.path.join(settings.DATA_ENTRY_PATH, name)

        entry.started = True
        entry.finished = False
        entry.nets_message = entry.users_message = entry.posts_message = entry.comments_message = \
            entry.likes_message = entry.reshares_message = entry.friends_message = ''
        entry.nets_progress = entry.users_progress = entry.posts_progress = entry.comments_progress = \
            entry.likes_progress = entry.reshares_progress = entry.friends_progress = 0
        entry.save()

        net_ids = run_entry(path, entry)
        for net_id in net_ids:
            net = SocialNet.objects.get(id=net_id)
            logger.info('============= text processing for net "%s"' % net.name)
            TextProcessing().process(net, include_comments=True)

        entry.finished = True
        entry.save()

        res = {'ok': True}
    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')


def run_entry(data_path, entry_obj):
    logger.info('creating nets ...')
    net_id_map = run_nets_entry(data_path, entry_obj)
    logger.info('creating users ...')
    user_id_map = run_users_entry(data_path, entry_obj, net_id_map)
    logger.info('creating friendships ...')
    run_friend_entry(data_path, entry_obj, user_id_map)
    logger.info('creating posts ...')
    post_id_map = run_posts_entry(data_path, entry_obj, user_id_map)
    logger.info('creating comments ...')
    run_comments_entry(data_path, entry_obj, user_id_map, post_id_map)
    logger.info('creating likes ...')
    run_likes_entry(data_path, entry_obj, user_id_map, post_id_map)
    del user_id_map
    logger.info('creating reshares ...')
    run_reshares_entry(data_path, entry_obj, post_id_map)

    net_ids = net_id_map.values()
    return net_ids


def run_nets_entry(data_path, entry_obj):
    net_id_map = {}
    file_path = os.path.join(data_path, 'nets.csv')
    if os.path.exists(file_path):
        with open(file_path) as csvfile:
            try:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                lines = sum(1 for _ in reader)
                step = max(1, lines / 20)
                csvfile.seek(0)
                count = 0
                for row in reader:
                    count += 1
                    try:
                        id, name, icon = row
                    except ValueError:
                        break
                    try:
                        net = SocialNet.objects.get(name=name)
                        net.icon = icon
                        net.save()
                    except ObjectDoesNotExist:
                        net = SocialNet.objects.create(name=name, icon=icon)
                    net_id_map[id] = net.id
                    if count % step == 0:
                        entry_obj.nets_progress = int(float(count) / lines * 100)
                        entry_obj.save()
                entry_obj.nets_progress = 100
                entry_obj.save()
            except:
                entry_obj.nets_message = u'فرمت داده شبکه اجتماعی استاندارد نیست'
    else:
        entry_obj.nets_message = u'داده شبکه اجتماعی موجود نیست'
    return net_id_map


def run_users_entry(data_path, entry_obj, net_id_map):
    user_id_map = {}
    file_path = os.path.join(data_path, 'users.csv')
    if os.path.exists(file_path):
        users = {
            net_id: {user.username: user.id for user in UserAccount.objects.filter(social_net_id=net_id).iterator()}
            for net_id in net_id_map.values()
        }

        with open(file_path) as csvfile:
            try:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                lines = sum(1 for _ in reader)
                step = max(1, lines / 20)
                csvfile.seek(0)
                count = 0
                for row in reader:
                    count += 1
                    try:
                        id, net_id, username, first_name, last_name, gender, birth_loc, location, language, bio, \
                        start_datetime, exit_datetime, avatar = row
                    except ValueError:
                        break
                    start_datetime = str_to_datetime(start_datetime) if start_datetime else None
                    exit_datetime = str_to_datetime(exit_datetime) if exit_datetime else None
                    gender = gender == '1' if gender else None
                    net_db_id = net_id_map[net_id]
                    if username not in users[net_db_id]:
                        person = Person.objects.create(first_name=first_name, last_name=last_name, gender=gender,
                                                       birth_loc=birth_loc, location=location, language=language,
                                                       bio=bio)
                        user = UserAccount.objects.create(person=person, social_net_id=net_id_map[net_id],
                                                          username=username,
                                                          start_datetime=start_datetime, exit_datetime=exit_datetime,
                                                          avatar=avatar)
                        user_id = user.id
                    else:
                        user_id = users[net_db_id][username]
                    user_id_map[id] = user_id
                    if count % step == 0:
                        entry_obj.users_progress = int(float(count) / lines * 100)
                        entry_obj.save()
                entry_obj.users_progress = 100
                entry_obj.save()
            except KeyError:
                entry_obj.users_message = u'شبکه اجتماعی ارجاع داده شده موجود نیست'
            except:
                entry_obj.users_message = u'فرمت داده کاربران استاندارد نیست'
    else:
        entry_obj.users_message = u'داده کاربران موجود نیست'
    entry_obj.save()
    return user_id_map


def run_friend_entry(data_path, entry_obj, user_id_map):
    file_path = os.path.join(data_path, 'friendships.csv')
    if os.path.exists(file_path):
        with open(file_path) as csvfile:
            try:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                lines = sum(1 for _ in reader)
                step = max(1, lines / 20)
                csvfile.seek(0)
                count = 0
                friendships = []
                for row in reader:
                    count += 1
                    try:
                        user_id, friend_user_id, start_datetime, end_datetime = row
                    except ValueError:
                        break
                    start_datetime = str_to_datetime(start_datetime) if start_datetime else None
                    end_datetime = str_to_datetime(end_datetime) if end_datetime else None
                    friendships.append(Friendship(user1_id=user_id_map[user_id], user2_id=user_id_map[friend_user_id],
                                                  start_datetime=start_datetime, end_datetime=end_datetime))
                    if count % step == 0:
                        Friendship.objects.bulk_create(friendships)
                        friendships = []
                        entry_obj.friends_progress = int(float(count) / lines * 100)
                        entry_obj.save()
                Friendship.objects.bulk_create(friendships)
                entry_obj.friends_progress = 100
                entry_obj.save()
            except KeyError:
                entry_obj.friends_message = u'کاربر ارجاع داده شده موجود نیست'
            except:
                entry_obj.friends_message = u'فرمت داده دوستی‌ها استاندارد نیست'
    else:
        entry_obj.friends_message = u'داده دوستی‌ها موجود نیست'
    entry_obj.save()
    return user_id_map


def run_posts_entry(data_path, entry_obj, user_id_map):
    post_id_map = {}
    file_path = os.path.join(data_path, 'posts.csv')
    if os.path.exists(file_path):
        with open(file_path) as csvfile:
            try:
                csv.field_size_limit(500 * 1024 * 1024)
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                lines = sum(1 for _ in reader)
                step = max(1, lines / 20)
                csvfile.seek(0)
                count = 0
                for row in reader:
                    count += 1
                    try:
                        id, user_id, dt, text = row
                    except ValueError:
                        break
                    dt = str_to_datetime(dt) if dt else None
                    post = Post.objects.create(author_id=user_id_map[user_id], datetime=dt, text=text)
                    post_id_map[id] = post.id
                    if count % step == 0:
                        entry_obj.posts_progress = int(float(count) / lines * 100)
                        entry_obj.save()
                    if count % 10000 == 0:
                        logger.info('%d posts done' % count)
                entry_obj.posts_progress = 100
                entry_obj.save()
            except KeyError:
                entry_obj.posts_message = u'کاربر ارجاع داده شده موجود نیست'
            except:
                entry_obj.posts_message = u'فرمت داده پست‌ها استاندارد نیست'
                print traceback.format_exc()
    else:
        entry_obj.posts_message = u'داده پست‌ها موجود نیست'
    entry_obj.save()
    return post_id_map


def run_comments_entry(data_path, entry_obj, user_id_map, post_id_map):
    file_path = os.path.join(data_path, 'comments.csv')
    if os.path.exists(file_path):
        with open(file_path) as csvfile:
            try:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                lines = sum(1 for _ in reader)
                step = max(1, lines / 20)
                csvfile.seek(0)
                count = 0
                comments = []
                for row in reader:
                    count += 1
                    try:
                        user_id, post_id, dt, text = row
                    except ValueError:
                        break
                    dt = str_to_datetime(dt) if dt else None
                    comments.append(
                        Comment(user_id=user_id_map[user_id], post_id=post_id_map[post_id], datetime=dt, text=text))
                    if count % step == 0:
                        Comment.objects.bulk_create(comments)
                        comments = []
                        entry_obj.comments_progress = int(float(count) / lines * 100)
                        entry_obj.save()
                Comment.objects.bulk_create(comments)
                entry_obj.comments_progress = 100
                entry_obj.save()
            except KeyError:
                entry_obj.comments_message = u'کاربر یا پست ارجاع داده شده موجود نیست'
            except:
                entry_obj.comments_message = u'فرمت داده نظرها استاندارد نیست'
    else:
        entry_obj.comments_message = u'داده نظرها موجود نیست'
    entry_obj.save()


def run_likes_entry(data_path, entry_obj, user_id_map, post_id_map):
    file_path = os.path.join(data_path, 'likes.csv')
    if os.path.exists(file_path):
        with open(file_path) as csvfile:
            try:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                lines = sum(1 for _ in reader)
                step = max(1, lines / 20)
                csvfile.seek(0)
                count = 0
                likes = []
                for row in reader:
                    count += 1
                    try:
                        user_id, post_id, dt = row
                    except ValueError:
                        break
                    dt = str_to_datetime(dt) if dt else None
                    likes.append(Like(user_id=user_id_map[user_id], post_id=post_id_map[post_id], datetime=dt))
                    if count % step == 0:
                        Like.objects.bulk_create(likes)
                        likes = []
                        entry_obj.likes_progress = int(float(count) / lines * 100)
                        entry_obj.save()
                Like.objects.bulk_create(likes)
                entry_obj.likes_progress = 100
                entry_obj.save()
            except KeyError:
                entry_obj.likes_message = u'کاربر یا پست ارجاع داده شده موجود نیست'
            except:
                entry_obj.likes_message = u'فرمت داده پسندها استاندارد نیست'
    else:
        entry_obj.likes_message = u'داده پسندها موجود نیست'
    entry_obj.save()


def run_reshares_entry(data_path, entry_obj, post_id_map):
    file_path = os.path.join(data_path, 'reshares.csv')
    if os.path.exists(file_path):
        with open(file_path) as csvfile:
            try:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                lines = sum(1 for _ in reader)
                step = max(1, lines / 20)
                csvfile.seek(0)
                count = 0
                reshares = []
                for row in reader:
                    count += 1
                    try:
                        post_id, original_post_id = row
                    except ValueError:
                        break
                    post = Post.objects.get(id=post_id_map[post_id])
                    ref_post = Post.objects.get(id=post_id_map[original_post_id])
                    reshares.append(
                        Reshare.objects.create(post=post, reshared_post=ref_post, user_id=post.author_id,
                                               ref_user_id=ref_post.author_id, datetime=post.datetime,
                                               ref_datetime=ref_post.datetime))
                    if count % step == 0:
                        Reshare.objects.bulk_create(reshares)
                        reshares = []
                        entry_obj.reshares_progress = int(float(count) / lines * 100)
                        entry_obj.save()
                Reshare.objects.bulk_create(reshares)
                entry_obj.reshares_progress = 100
                entry_obj.save()
            except KeyError:
                entry_obj.reshares_message = u'پست ارجاع داده شده موجود نیست'
            except:
                entry_obj.reshares_message = u'فرمت داده بازنشرها استاندارد نیست'
    else:
        entry_obj.reshares_message = u'داده بازنشرها موجود نیست'
    entry_obj.save()
