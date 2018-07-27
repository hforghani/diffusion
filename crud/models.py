# -*- coding: utf-8 -*-
from django.db import models
from django.db.models import Q


class SocialNet(models.Model):
    name = models.CharField(u'نام', max_length=50)
    icon = models.CharField(u'نشان', max_length=200)

    class Meta:
        verbose_name, verbose_name_plural = u'شبکه اجتماعی', u'شبکه‌های اجتماعی'

    def __unicode__(self):
        return self.name


class UserAccount(models.Model):
    social_net = models.ForeignKey(SocialNet, verbose_name=u'شبکه اجتماعی', on_delete=models.CASCADE)
    username = models.CharField(u'نام کاربری', max_length=100)
    friends_count = models.IntegerField(u'تعداد دوستان', null=True, blank=True)
    start_datetime = models.DateTimeField(u'زمان آغاز', null=True, blank=True)
    exit_datetime = models.DateTimeField(u'زمان پایان', null=True, blank=True)
    avatar = models.CharField(u'عکس آواتار', max_length=200, null=True, blank=True)

    class Meta:
        verbose_name, verbose_name_plural = u'حساب کاربری', u'حساب‌های کاربری'

    def __unicode__(self):
        res = self.username
        if self.person:
            res += ' (%s %s)' % (self.person.first_name, self.person.last_name)
        return res

    def get_friends(self):
        return Friendship.objects.filter(Q(user1=self) | Q(user2=self))

    def get_dict(self):
        u = {
            'id': self.id,
            'social_net_id': self.social_net.id,
            'username': self.username,
            'avatar': self.avatar,
        }
        return u


class Friendship(models.Model):
    user1 = models.ForeignKey(UserAccount, verbose_name=u'کاربر 1', related_name='+', db_index=True,
                              on_delete=models.CASCADE)
    user2 = models.ForeignKey(UserAccount, verbose_name=u'کاربر 2', related_name='+', db_index=True,
                              on_delete=models.CASCADE)
    start_datetime = models.DateTimeField(u'زمان آغاز', null=True, blank=True)
    end_datetime = models.DateTimeField(u'زمان پایان', null=True, blank=True)

    class Meta:
        verbose_name, verbose_name_plural = u'رابطه دوستی', u'روابط دوستی'


class Post(models.Model):
    author = models.ForeignKey(UserAccount, verbose_name=u'نویسنده', on_delete=models.CASCADE)
    datetime = models.DateTimeField(u'زمان', null=True, blank=True, db_index=True)
    text = models.TextField(u'متن', null=True, blank=True)
    url = models.CharField(u'آدرس', null=True, blank=True, max_length=100, db_index=True)

    class Meta:
        verbose_name, verbose_name_plural = u'پست', u'پست‌ها'

    def __unicode__(self):
        return u'پست توسط %s در %s' % (self.author.username, str(self.datetime))

    def get_dict(self):
        out = {
            'id': self.id,
            'text': self.text,
            'datetime': str(self.datetime),
            'author_username': self.author.username
        }
        if self.author.person:
            out.update({
                'author_first_name': self.author.person.first_name,
                'author_last_name': self.author.person.last_name
            })
        return out


class Reshare(models.Model):
    post = models.ForeignKey(Post, verbose_name=u'پست', related_name='parents', on_delete=models.CASCADE)
    reshared_post = models.ForeignKey(Post, verbose_name=u'پست مرجع', related_name='children', null=True,
                                      on_delete=models.CASCADE)
    user = models.ForeignKey(UserAccount, verbose_name=u'کاربر', null=True, blank=True, related_name='parent_reshares',
                             on_delete=models.CASCADE)
    ref_user = models.ForeignKey(UserAccount, verbose_name=u'کاربر مرجع', null=True, blank=True,
                                 related_name='children_reshares', on_delete=models.CASCADE)
    datetime = models.DateTimeField(u'زمان', null=True, blank=True, db_index=True)
    ref_datetime = models.DateTimeField(u'زمان پست مرجع', null=True, blank=True)

    class Meta:
        verbose_name, verbose_name_plural = u'بازنشر', u'بازنشرها'


class Meme(models.Model):
    text = models.TextField(u'متن', null=True, blank=True)
    count = models.IntegerField(u'تعداد انتشار', null=True, blank=True)
    first_time = models.DateTimeField(u'اولین انتشار', null=True, blank=True)
    last_time = models.DateTimeField(u'آخرین انتشار', null=True, blank=True)
    depth = models.IntegerField(u'عمق', null=True, blank=True)

    class Meta:
        verbose_name, verbose_name_plural = u'محتوای جریان‌ساز', u'محتواهای جریان‌ساز'


class PostMeme(models.Model):
    post = models.ForeignKey(Post, verbose_name=u'پست', db_index=True, on_delete=models.CASCADE)
    meme = models.ForeignKey(Meme, verbose_name=u'محتوای جریان‌ساز', db_index=True, on_delete=models.CASCADE)
