from django.conf.urls import patterns, url

urlpatterns = patterns('',
                       url(r'ajax/$', 'crud.ajax.main', name='crud/ajax'),
                       url(r'ajax/education/$', 'crud.ajax.person_education', name='crud/ajax_edu'),
                       url(r'ajax/work/$', 'crud.ajax.person_work', name='crud/ajax_work'),
                       url(r'ajax/entry/stat/$', 'crud.ajax.entry_stat', name='crud/entry_stat'),
                       url(r'ajax/entry/start/$', 'crud.ajax.start_entry_process', name='crud/entry_start'),

                       url(r'entry/list/$', 'crud.views.entry_list', name='crud/entry_list'),
                       url(r'entry/$', 'crud.views.entry', name='crud/entry'),
)
