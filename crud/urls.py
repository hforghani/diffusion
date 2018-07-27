from django.conf.urls import url
from crud import ajax

urlpatterns = [url(r'ajax/$', ajax.main, name='crud/ajax'),
]
