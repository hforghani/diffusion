from django.conf.urls import patterns, url

urlpatterns = patterns('',
                       url(r'ajax/$', 'crud.ajax.main', name='crud/ajax'),
)
