import os
from django.conf.urls import include, url
from django.conf import settings
from django.views.static import serve
from accounts import views as accounts_view

# Uncomment the next two lines to enable the admin:
from django.contrib import admin

admin.autodiscover()

urlpatterns = [
    url(r'^crud/', include('crud.urls')),
    url(r'^diffusion/', include('cascade.urls')),
    url(r'^$', accounts_view.home, name='home'),

    # Uncomment the admin/doc line below to enable admin documentation:
    url(r'^admin/doc/', include('django.contrib.admindocs.urls')),
    # Uncomment the next line to enable the admin:
    url(r'^admin/', admin.site.urls),
]

if settings.SERVE_STATIC_FILES:
    urlpatterns += [
        url(r'^bower_components/(?P<path>.*)$', serve,
            {'document_root': os.path.join(settings.STATIC_ROOT, 'bower_components'), }),
        url(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATIC_ROOT, }),
        url(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT, }),
    ]
