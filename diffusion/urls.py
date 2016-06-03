import os
from django.conf.urls import patterns, include, url
from django.conf import settings
from django.views.generic import RedirectView
from django.views.static import serve

# Uncomment the next two lines to enable the admin:
from django.contrib import admin

admin.autodiscover()

urlpatterns = patterns('',
                       url(r'^crud/', include('crud.urls')),
                       url(r'^diffusion/', include('cascade.urls')),
                       url(r'^$', 'accounts.views.home', name='home'),

                       # Uncomment the admin/doc line below to enable admin documentation:
                       url(r'^admin/doc/', include('django.contrib.admindocs.urls')),
                       # Uncomment the next line to enable the admin:
                       url(r'^admin/', include(admin.site.urls)),
)

if settings.SERVE_STATIC_FILES:
    urlpatterns += patterns(
        '',
        url(r'^bower_components/(?P<path>.*)$', serve,
            {'document_root': os.path.join(settings.STATIC_ROOT, 'bower_components'), }),
        url(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATIC_ROOT, }),
        url(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT, }),
    )
