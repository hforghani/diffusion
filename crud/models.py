# -*- coding: utf-8 -*-
import json
import os
from django.conf import settings
from django.db import models
from django.db.models import Q


class SocialNet(models.Model):
    name = models.CharField(u'نام', max_length=50)
    icon = models.CharField(u'نشان', max_length=200)

    class Meta:
        verbose_name, verbose_name_plural = u'شبکه اجتماعی', u'شبکه‌های اجتماعی'

    def __unicode__(self):
        return self.name


class Education(models.Model):
    type = models.CharField(u'نوع', max_length=50)
    field = models.CharField(u'رشته', max_length=50)
    degree = models.CharField(u'مدرک', max_length=50)
    school = models.CharField(u'دانشگاه', max_length=50)
    start_year = models.IntegerField(u'سال آغاز')
    end_year = models.IntegerField(u'سال پایان')

    class Meta:
        verbose_name, verbose_name_plural = u'تحصیلات', u'تحصیلات'


class Work(models.Model):
    place = models.CharField(u'مکان', max_length=50)
    position = models.CharField(u'سمت', max_length=50)
    start_date = models.DateField(u'تاریخ آغاز', max_length=50)
    end_date = models.DateField(u'تاریخ پایان', max_length=50)

    class Meta:
        verbose_name, verbose_name_plural = u'شغل', u'شغل‌ها'


LANGUAGES = [('fa', u'فارسی'),
             ('en', u'انگلیسی'),
             ('ar', u'عربی'),
             ('tr', u'ترکی'),
             ('fr', u'فرانسوی'),
             ('es', u'اسپانیایی'),
             ('it', u'ایتالیایی'),
             ('ch', u'چینی'),
             ('ja', u'ژاپنی'),
             ('ko', u'کره‌ای'),
             ('he', u'عبری')]

GENDERS = [
    (True, 'مرد'),
    (False, 'زن'),
    (None, 'نامعین')
]


class Person(models.Model):
    first_name = models.CharField(u'نام', max_length=50, null=True, blank=True)
    last_name = models.CharField(u'نام خانوادگی', max_length=50, null=True, blank=True)
    gender = models.NullBooleanField(u'جنسیت', choices=GENDERS, null=True, blank=True)
    birth_loc = models.CharField(u'محل تولد', max_length=50, null=True, blank=True)
    location = models.CharField(u'محل سکونت', max_length=50, null=True, blank=True)
    language = models.CharField(u'زبان', max_length=50, choices=LANGUAGES, null=True, blank=True)
    educations = models.ManyToManyField(Education, verbose_name=u'تحصیلات', null=True, blank=True)
    works = models.ManyToManyField(Work, verbose_name=u'شغل', null=True, blank=True)
    bio = models.TextField(u'مشخصات', null=True, blank=True)

    class Meta:
        verbose_name, verbose_name_plural = u'فرد', u'افراد'

    def __unicode__(self):
        return '%s %s' % (self.first_name, self.last_name)


class UserAccount(models.Model):
    social_net = models.ForeignKey(SocialNet, verbose_name=u'شبکه اجتماعی')
    person = models.ForeignKey(Person, verbose_name=u'فرد', null=True, blank=True)
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
            'person_id': self.person.id if self.person else None,
        }
        if self.person:
            u['first_name'] = self.person.first_name
            u['last_name'] = self.person.last_name
        return u


class Friendship(models.Model):
    user1 = models.ForeignKey(UserAccount, verbose_name=u'کاربر 1', related_name='+', db_index=True)
    user2 = models.ForeignKey(UserAccount, verbose_name=u'کاربر 2', related_name='+', db_index=True)
    start_datetime = models.DateTimeField(u'زمان آغاز', null=True, blank=True)
    end_datetime = models.DateTimeField(u'زمان پایان', null=True, blank=True)

    class Meta:
        verbose_name, verbose_name_plural = u'رابطه دوستی', u'روابط دوستی'


class Post(models.Model):
    author = models.ForeignKey(UserAccount, verbose_name=u'نویسنده')
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


class Like(models.Model):
    post = models.ForeignKey(Post, verbose_name=u'پست')
    user = models.ForeignKey(UserAccount, verbose_name=u'کاربر')
    datetime = models.DateTimeField(u'زمان', null=True, blank=True)

    class Meta:
        verbose_name, verbose_name_plural = u'پسند', u'پسندها'


class Comment(models.Model):
    post = models.ForeignKey(Post, verbose_name=u'پست')
    user = models.ForeignKey(UserAccount, verbose_name=u'کاربر')
    datetime = models.DateTimeField(u'زمان', null=True, blank=True)
    text = models.TextField(u'متن', null=True, blank=True)

    class Meta:
        verbose_name, verbose_name_plural = u'نظر', u'نظرها'


class Reshare(models.Model):
    post = models.ForeignKey(Post, verbose_name=u'پست', related_name='parents')
    reshared_post = models.ForeignKey(Post, verbose_name=u'پست مرجع', related_name='children', null=True)
    user = models.ForeignKey(UserAccount, verbose_name=u'کاربر', null=True, blank=True, related_name='parent_reshares')
    ref_user = models.ForeignKey(UserAccount, verbose_name=u'کاربر مرجع', null=True, blank=True,
                                 related_name='children_reshares')
    datetime = models.DateTimeField(u'زمان', null=True, blank=True, db_index=True)
    ref_datetime = models.DateTimeField(u'زمان پست مرجع', null=True, blank=True)

    class Meta:
        verbose_name, verbose_name_plural = u'بازنشر', u'بازنشرها'


class Group(models.Model):
    name = models.CharField(u'نام', max_length=50)
    type = models.CharField(u'نوع', max_length=50, null=True, blank=True)
    social_net = models.ForeignKey(SocialNet, verbose_name=u'شبکه اجتماعی', null=True, blank=True)
    avatar = models.CharField(u'عکس آواتار', max_length=200, null=True, blank=True)


class GroupMembership(models.Model):
    group = models.ForeignKey(Group, verbose_name=u'گروه')
    user = models.ForeignKey(UserAccount, verbose_name=u'کاربر')
    datetime = models.DateTimeField(u'زمان عضویت', null=True, blank=True)

    class Meta:
        verbose_name, verbose_name_plural = u'عضویت گروه', u'عضویت گروه'


class AnalysisName(object):
    NEIGHBORS = 'neighbor'
    CENTRALITY = 'cent'
    COMM_DET = 'commdet'
    COMM_EVOL = 'evol'
    COMM_EVOL_PREDICT = 'evolpred'
    FLOW = 'flow'
    KEYWORDS = 'kw'
    KW_CLUSTER = 'kwclust'
    DOC_CLUSTER = 'docclust'


class Project(models.Model):
    name = models.CharField(u'نام', max_length=50)

    _save_dir = os.path.join(settings.BASEPATH, 'resources', 'results')

    class Meta:
        verbose_name, verbose_name_plural = u'پروژه', u'پروژه‌ها'

    def __unicode__(self):
        return self.name

    def _file_path(self, analysis_name, **kwargs):
        parts = [analysis_name, str(self.id)]
        valid_keys = ['criteria', 'supervised', 'method', 'from_dt', 'to_dt', 'period']
        if kwargs:
            for key in valid_keys:
                if key in kwargs:
                    parts.append(str(kwargs[key]))
        fname = ('%s.json' % '-'.join(parts)).replace(':', '-').replace(' ', '-')
        return os.path.join(self._save_dir, fname)

    def save_results(self, res, analysis_name, **kwargs):
        if not os.path.exists(self._save_dir):
            os.mkdir(self._save_dir)
        json.dump(res, open(self._file_path(analysis_name, **kwargs), 'w'))

    def load_results(self, analysis_name, **kwargs):
        path = self._file_path(analysis_name, **kwargs)
        if os.path.exists(path):
            return json.load(open(path))
        else:
            return None


class ProjectMembership(models.Model):
    project = models.ForeignKey(Project, verbose_name=u'پروژه')
    user = models.ForeignKey(UserAccount, verbose_name=u'کاربر')

    class Meta:
        verbose_name, verbose_name_plural = u'عضویت پروژه', u'عضویت پروژه'


class Hashtag(models.Model):
    name = models.CharField(u'نام', max_length=255)

    class Meta:
        verbose_name, verbose_name_plural = u'هشتگ', u'هشتگ‌ها'

    def __unicode__(self):
        return self.name


class HashtagPost(models.Model):
    hashtag = models.ForeignKey(Hashtag, verbose_name=u'هشتگ')
    post = models.ForeignKey(Post, verbose_name=u'پست')

    class Meta:
        verbose_name, verbose_name_plural = u'هشتگ-پست', u'هشتگ-پست‌ها'


class Keyword(models.Model):
    name = models.CharField(u'کلیدواژه', max_length=50, unique=True)
    idf = models.FloatField(null=True, blank=True)

    class Meta:
        verbose_name, verbose_name_plural = u'کلیدواژه', u'کلیدواژه‌ها'

    def __unicode__(self):
        return self.name

    def get_dict(self):
        return {'id': self.id, 'name': self.name}


class TermFrequency(models.Model):
    post = models.ForeignKey(Post, verbose_name=u'پست')
    term = models.ForeignKey(Keyword, verbose_name=u'کلیدواژه')
    frequency = models.IntegerField(u'تکرار', default=0)

    class Meta:
        verbose_name, verbose_name_plural = u'تکرار کلمه پست', u'تکرار کلمات پست‌ها'


class UserTermFreq(models.Model):
    user = models.ForeignKey(UserAccount, verbose_name=u'پست')
    term = models.ForeignKey(Keyword, verbose_name=u'کلیدواژه')
    frequency = models.IntegerField(u'تکرار', default=0)
    from_datetime = models.DateTimeField(u'از', null=True)
    to_datetime = models.DateTimeField(u'تا', null=True)

    class Meta:
        verbose_name, verbose_name_plural = u'تکرار کلمه کاربر', u'تکرار کلمات کاربران'


class TimeStep(models.Model):
    datetime = models.DateTimeField(u'زمان')


class CoOccurrence(models.Model):
    term1 = models.ForeignKey(Keyword, verbose_name=u'کلیدواژه 1', related_name='+')
    term2 = models.ForeignKey(Keyword, verbose_name=u'کلیدواژه 2', related_name='+')
    count = models.IntegerField(u'تعداد', default=0, db_index=True)

    class Meta:
        verbose_name, verbose_name_plural = u'وقوع هم‌زمان', u'اطلاعات وقوع هم‌زمان'


class Topic(models.Model):
    name = models.CharField(u'نام', max_length=50)
    users = models.ManyToManyField(UserAccount, verbose_name=u'کاربران')

    class Meta:
        verbose_name, verbose_name_plural = u'موضوع', u'موضوع‌ها'


class Meme(models.Model):
    text = models.TextField(u'متن', null=True, blank=True)
    count = models.IntegerField(u'تعداد انتشار', null=True, blank=True)
    first_time = models.DateTimeField(u'اولین انتشار', null=True, blank=True)
    last_time = models.DateTimeField(u'آخرین انتشار', null=True, blank=True)

    class Meta:
        verbose_name, verbose_name_plural = u'محتوای جریان‌ساز', u'محتواهای جریان‌ساز'


class PostMeme(models.Model):
    post = models.ForeignKey(Post, verbose_name=u'پست', db_index=True)
    meme = models.ForeignKey(Meme, verbose_name=u'محتوای جریان‌ساز', db_index=True)


class DiffusionParam(models.Model):
    sender = models.ForeignKey(UserAccount, verbose_name=u'فرستنده', related_name='params_from')
    receiver = models.ForeignKey(UserAccount, verbose_name=u'گیرنده', related_name='params_to')
    weight = models.FloatField(verbose_name=u'وزن', default=0)
    delay = models.FloatField(verbose_name=u'تأخیر', default=0)  # in days


class DataEntry(models.Model):
    name = models.CharField(u'نام', max_length=200)
    nets_progress = models.IntegerField(u'پیشرفت شبکه اجتماعی', default=0)
    users_progress = models.IntegerField(u'پیشرفت کاربران', default=0)
    posts_progress = models.IntegerField(u'پیشرفت پست‌ها', default=0)
    comments_progress = models.IntegerField(u'پیشرفت نظرها', default=0)
    likes_progress = models.IntegerField(u'پیشرفت پسندها', default=0)
    reshares_progress = models.IntegerField(u'پیشرفت بازنشرها', default=0)
    friends_progress = models.IntegerField(u'پیشرفت دوستی‌ها', default=0)
    nets_message = models.CharField(u'پیغام شبکه اجتماعی', max_length=500)
    users_message = models.CharField(u'پیغام کاربران', max_length=500)
    posts_message = models.CharField(u'پیغام پست‌ها', max_length=500)
    comments_message = models.CharField(u'پیغام نظرها', max_length=500)
    likes_message = models.CharField(u'پیغام پسندها', max_length=500)
    reshares_message = models.CharField(u'پیغام بازنشرها', max_length=500)
    friends_message = models.CharField(u'پیغام دوستی‌ها', max_length=500)
    started = models.BooleanField(u'آغازشده', default=False)
    finished = models.BooleanField(u'تمام‌شده', default=False)
