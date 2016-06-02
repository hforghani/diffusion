# -*- coding: utf-8 -*-
import json
import logging
import time

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Count, Min
from django.http import HttpResponse

from crud.models import Post, UserAccount, Meme
from diffusion.models import Generation, CascadePredictor, CascadeTree
from utils.time_utils import timeline_data, str_to_datetime, DT_FORMAT


logger = logging.getLogger('social.diffusion.ajax')


@login_required
def memes(request):
    """
    GET parameters:
        from: start date-time
        to: end date-time
        query: some text to query in post meme texts
        page_size: number of page rows
        page: page number
    output:
        {
        memes: [meme1, meme2, ...],
        total: <total_count>
        }
    """
    res = []
    if request.method == 'GET':
        from_str = request.GET.get('from', None)
        to_str = request.GET.get('to', None)
        text = request.GET.get('text', None)
        page_size = int(request.GET.get('page_size', 10))
        page = int(request.GET.get('page', 1))

        # Convert 'from' and 'to' strings to datetime.
        from_dt = None
        to_dt = None
        if from_str:
            from_dt = str_to_datetime(from_str)
        if to_str:
            to_dt = str_to_datetime(to_str)

        # Query memes with any post after between 'to' and 'from' datetimes.
        queryset = Meme.objects.all()
        if from_dt:
            queryset = queryset.filter(first_time__gt=from_dt)
        if to_dt:
            queryset = queryset.filter(last_time__lt=to_dt)
        if text:
            #TODO: best results to top
            #words = re.compile(r'\W+', re.UNICODE).split(text)
            queryset = queryset.filter(text__icontains=text)
        queryset = queryset.order_by('-count')

        # Paginate items.
        paginator = Paginator(queryset, page_size)
        if page > paginator.num_pages:
            page = paginator.num_pages
        if page < 1:
            page = 1
        objects = paginator.page(page).object_list

        res = {
            'memes': [
                {
                    'id': meme.id,
                    'text': meme.text,
                    'count': meme.count,
                    'first_time': meme.first_time.strftime(DT_FORMAT) if meme.first_time else None,
                }
                for meme in objects
            ],
            'total': queryset.count()
        }
    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')


@login_required
def first_user(request):
    res = {'ok': False}
    if request.method == 'GET':
        if not 'id' in request.GET:
            raise ValueError('no meme id given')
        meme_id = request.GET['id']
        logger.info('meme users bubble started ...')
        start = time.time()
        try:
            min_time = Meme.objects.get(id=meme_id).postmeme_set.aggregate(min=Min('post__datetime'))['min']
            posts = Post.objects.filter(datetime=min_time)
            user = None
            for post in posts:
                if post.postmeme_set.filter(meme=meme_id).exists():
                    user = post.author
                    break
            res = user.get_dict()
        except IndexError:
            res = None
        logger.info('first user process time = %f' % (time.time() - start))

    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')


@login_required
def users_bubble(request):
    """
    Returns the authors with most contribution in a meme and the amount of their contribution.
    GET parameters:
        id: meme id
    output:
        [
        {id:<user1_id>, username:<user1_username>, ..., size:<user1_size_of_bubble>},
        ...
        ]
    """
    res = {'ok': False}
    if request.method == 'GET':
        if not 'id' in request.GET:
            raise ValueError('no meme id given')
        meme_id = request.GET['id']
        count = request.GET.get('count', 10)

        logger.info('meme users bubble started ...')
        start = time.time()
        users = UserAccount.objects.filter(post__postmeme__meme=meme_id).distinct().annotate(Count('post')) \
                    .order_by('-post__count')[:count]
        res = []
        for user in users:
            user_info = user.get_dict()
            user_info['size'] = user.post__count
            res.append(user_info)
        res = sorted(res, key=lambda obj: obj['size'], reverse=True)
        logger.info('meme users bubble process time = %f' % (time.time() - start))
    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')


@login_required
def timeline(request):
    """
    GET parameters:
        id: meme id
    output:
        {
        scale: <time tick scale>,
        counts: [{datetime:<dt1>, count:<count1>}, ...]
        }
    """
    res = {'ok': False}
    if request.method == 'GET':
        if not 'id' in request.GET:
            raise ValueError('no meme id given')
        meme_id = request.GET['id']

        logger.info('meme time-line started ...')
        start = time.time()
        datetimes = list(
            Post.objects.filter(postmeme__meme=meme_id, datetime__isnull=False).distinct().values_list('datetime',
                                                                                                       flat=True))
        if datetimes:
            res = timeline_data(datetimes)
        logger.info('meme time-line process time = %f' % (time.time() - start))

    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')


@login_required
def cascade_tree(request):
    """
    GET parameters:
        id: meme id
    output:
        [
            {user: <user_spec>,
            datetime: <post_datetime>,
            post_id: <post id>,
            children: [
                { ... <receiver1_spec> ... },
                { ... <receiver2_spec> ... },
                ...
            ]},
            {...},
            ...
        ]
    """
    res = {'ok': False}
    if request.method == 'GET':
        start = time.time()
        logger.info('cascade tree started ...')
        if not 'id' in request.GET:
            raise ValueError('no meme id given')
        meme_id = request.GET['id']
        meme = Meme.objects.get(id=meme_id)
        res = CascadeTree().extract_cascade(meme).get_dict()
        logger.info('cascade tree process time = %f' % (time.time() - start))

    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')


@login_required
def user_meme_activities(request):
    """
    GET parameters:
        meme_id: meme id
        user_id: user id
    output:
        [datetime1, datetime2, ...]
    """
    res = {'ok': False}
    if request.method == 'GET':
        try:
            user_id = request.GET['user_id']
        except KeyError:
            raise ValueError('no user id given')
        try:
            meme_id = request.GET['meme_id']
        except KeyError:
            raise ValueError('no meme id given')

        datetimes = Post.objects.filter(author=user_id, postmeme__meme=meme_id).order_by('datetime').values_list(
            'datetime', flat=True)
        res = [str(dt) for dt in datetimes]

    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')


@login_required
def generation(request):
    """
    input json parameters:
        meme_id: meme id
        generations: [
            {number: 1, users: [user1_id, user2_id, ...]},
            {number: 2, users: [user1_id, user2_id, ...]},
            ...
        ]
    output:
        [ {number: 1, result: {
                                first_post: {
                                    user: { ... user_spec ...},
                                    datetime: <post_datetime>,
                                },
                                last_post: {
                                    user: { ... user_spec ...},
                                    datetime: <post_datetime>,
                                },
                                avg_text: [
                                    {text: <some_sentences>, color: <black_or_green>},
                                    ...
                                ]
                            }
            },
            ...
        ]
    """
    res = {'ok': False}
    data = json.loads(request.body)
    if data:
        start = time.time()
        logger.info('generation started ...')

        meme_id = data['meme_id']
        res = []
        prev_gener = None
        for gener_data in data['generations']:
            num = gener_data['number']
            users_id = gener_data['users']
            posts = Post.objects.filter(postmeme__meme=meme_id, author__in=users_id).order_by('datetime')
            first = posts[0]
            last = posts.latest('datetime')

            gener = Generation(posts)
            if prev_gener:
                avg = gener.get_diff(prev_gener)
            else:
                avg = [{'color': 'black', 'text': gener.get_summary()}]

            res.append({'number': num, 'result': {
                'first_post': {
                    'user': first.author.get_dict(),
                    'datetime': first.datetime.strftime(DT_FORMAT)
                },
                'last_post': {
                    'user': last.author.get_dict(),
                    'datetime': last.datetime.strftime(DT_FORMAT)
                },
                'avg_text': avg
            }})
            prev_gener = gener
        logger.info('generation process time = %f' % (time.time() - start))

    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')


@login_required
def predict(request):
    """
    input json parameters:
        { tree: diffusion tree exactly same as output of web-service "tree".
          period: number of days in future }
    output:
        { tree: diffusion tree of the past and future with the same structure as the input.
          now: date-time of now }
    """
    res = {'ok': False}
    data = json.loads(request.body)
    if data:
        start = time.time()
        logger.info('cascade prediction started ...')
        tree = data['tree']

        initial_tree = CascadeTree(tree)
        tree = CascadePredictor(initial_tree).predict().get_dict()
        res = {'tree': tree, 'now': initial_tree.max_datetime().strftime(DT_FORMAT)}
        #res = {'tree': tree, 'now': now.strftime(DT_FORMAT)}
        logger.info('cascade tree prediction process time = %f' % (time.time() - start))

    return HttpResponse(json.dumps(res), content_type='application/json', mimetype='application/json')
