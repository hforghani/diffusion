from django.contrib.auth.decorators import login_required
from django.shortcuts import render_to_response
from django.template import RequestContext
from crud.models import Meme


@login_required
def main(request):
    return render_to_response('diffusion/main.html', {}, RequestContext(request))


@login_required
def meme_page(request):
    meme = None
    if request.method == 'GET':
        meme_id = request.GET.get('id', None)
        if not meme_id:
            raise ValueError('no meme id given')
        meme = Meme.objects.get(id=meme_id)
    return render_to_response('diffusion/profile.html', {'meme': meme}, RequestContext(request))
