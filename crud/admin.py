# -*- coding: utf-8 -*-

from django.contrib import admin
from django.contrib.sites.models import Site
from django.utils.safestring import mark_safe
from crud.models import *


def text_trunc(obj):
    if obj.text:
        if len(obj.text) > 100:
            return obj.text[:100] + ' ...'
        else:
            return obj.text
    else:
        return None


text_trunc.short_description = u'متن'


class FriendshipAdmin(admin.ModelAdmin):
    list_display = ('user1', 'user2', 'start_datetime')
    search_fields = ['user1__username', 'user1__person__first_name', 'user1__person__last_name',
                     'user2__username', 'user2__person__first_name', 'user2__person__last_name']
    readonly_fields = ['user1', 'user2']


class UserAccountAdmin(admin.ModelAdmin):
    list_display = ('username', 'social_net', 'first_name', 'last_name', 'avatar_img')
    search_fields = ['username', 'person__first_name', 'person__last_name']
    readonly_fields = ['person']

    def first_name(self, obj):
        return obj.person.first_name if obj.person else None

    def last_name(self, obj):
        return obj.person.last_name if obj.person else None

    def avatar_img(self, obj):
        return mark_safe('<img src="%s" style="width:20px">' % obj.avatar) if obj.avatar else None

    first_name.short_description = u'نام'
    last_name.short_description = u'نام خانوادگی'
    avatar_img.short_description = u'عکس آواتار'


class PersonAdmin(admin.ModelAdmin):
    list_display = ('first_name', 'last_name', 'gender', 'get_language_display')
    search_fields = ['first_name', 'last_name']


class PostAdmin(admin.ModelAdmin):
    list_display = ('author', 'datetime', text_trunc)
    search_fields = ['author__username', 'text']
    readonly_fields = ['author']


class LikeAdmin(admin.ModelAdmin):
    list_display = ('user', 'post', 'datetime')
    search_fields = ['user__username', 'post__author__username']
    readonly_fields = ['user']


class CommentAdmin(admin.ModelAdmin):
    list_display = ('user', 'post', 'datetime', 'text')
    search_fields = ['user__username', 'post__author__username']
    readonly_fields = ['user']


class ReshareAdmin(admin.ModelAdmin):
    list_display = ('post', 'reshared_post', 'datetime')
    search_fields = ['post__author__username', 'reshared_post__author__username']
    readonly_fields = ['post', 'reshared_post']


class TopicAdmin(admin.ModelAdmin):
    list_display = ('name', 'users_list')
    readonly_fields = ['users']

    def users_list(self, obj):
        return ', '.join([unicode(u) for u in obj.users.all()])

    users_list.short_description = u'کاربران'


class MemeAdmin(admin.ModelAdmin):
    list_display = (text_trunc, 'count', 'first_time')


class KeywordAdmin(admin.ModelAdmin):
    list_display = ('name', 'idf')
    search_fields = ['name']


admin.site.register(SocialNet)
admin.site.register(UserAccount, UserAccountAdmin)
admin.site.register(Person, PersonAdmin)
admin.site.register(Friendship, FriendshipAdmin)
admin.site.register(Post, PostAdmin)
admin.site.register(Like, LikeAdmin)
admin.site.register(Comment, CommentAdmin)
admin.site.register(Reshare, ReshareAdmin)
admin.site.register(Project)
admin.site.register(ProjectMembership)
admin.site.register(Hashtag)
admin.site.register(Topic, TopicAdmin)
admin.site.register(Meme, MemeAdmin)
admin.site.register(Keyword, KeywordAdmin)
admin.site.unregister(Site)