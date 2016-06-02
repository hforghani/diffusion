from django.conf.urls import patterns, url

urlpatterns = patterns('',
                       url(r'^$', 'diffusion.views.main', name='diffusion'),
                       url(r'meme/$', 'diffusion.views.meme_page', name='diffusion/meme'),

                       url(r'ajax/memes/$', 'diffusion.ajax.memes', name='diffusion/memes'),
                       url(r'ajax/first_user/$', 'diffusion.ajax.first_user', name='diffusion/first_user'),
                       url(r'ajax/users_bubble/$', 'diffusion.ajax.users_bubble', name='diffusion/users_bubble'),
                       url(r'ajax/timeline/$', 'diffusion.ajax.timeline', name='diffusion/timeline'),
                       url(r'ajax/tree/$', 'diffusion.ajax.cascade_tree', name='diffusion/tree'),
                       url(r'ajax/activities/$', 'diffusion.ajax.user_meme_activities', name='diffusion/activities'),
                       url(r'ajax/generation/$', 'diffusion.ajax.generation', name='diffusion/generation'),
                       url(r'ajax/predict/$', 'diffusion.ajax.predict', name='diffusion/predict'),

)
