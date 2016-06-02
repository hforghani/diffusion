# -*- coding: utf-8 -*-
import random
import string
import urllib
from bs4 import BeautifulSoup
import re
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.core.management.base import BaseCommand
from django.utils.timezone import get_default_timezone_name
import datetime
import jdatetime
import pytz
from crud.models import Person, UserAccount, Post, SocialNet
from utils.text_utils import unify_text
from utils.time_utils import shamsi_month_num


class Command(BaseCommand):
    help = 'Parse news html files and extracts text, author, and datetimes.'

    names = []

    def __init__(self):
        super(Command, self).__init__()
        with open(settings.MALE_FNAME) as f:
            content = f.read()
            self.names = content.strip().split('\n')
        with open(settings.FEMALE_FNAME) as f:
            content = f.read()
            self.names.extend(content.strip().split('\n'))

    def handle(self, *args, **options):
        irna_url = 'http://www.irna.ir/fa/NewsPage.aspx?zone=6&lm=Latest&area=0&title=884o19SRTQxg4OCGvmQl4J01bP2HgRWdDBD35VWuQpA%3d&lang=fa&pageno='
        pages = [
            ('irna', '%s1' % irna_url),
            ('irna', '%s2' % irna_url),
            ('irna', '%s3' % irna_url),
            ('irna', '%s4' % irna_url),
            ('irna', '%s5' % irna_url),
            ('irna', '%s6' % irna_url),
            ('alef', 'http://alef.ir/vtpjufwmsuqe.zf.html'),
            ('alef', 'http://alef.ir/vtpbrul0irhb.p15ruu.3e.html'),
            ('alef', 'http://alef.ir/vtpc2a,el2bq.8612aa.jy.html'),
            ('alef', 'http://alef.ir/vtpdy2h9ayt0.6qoy22.bm.html'),
            ('alef', 'http://alef.ir/vtpejb2p9jh8.ilnjbb.6k.html'),
            ('alef', 'http://alef.ir/vtpfwipvgw6d.a9vwii.z7.html'),
        ]
        for source, url in pages:
            f = urllib.urlopen(url)
            html = f.read()
            parsed_html = BeautifulSoup(html, 'html.parser')
            self.parse_arch_page(parsed_html, source)

    def parse_arch_page(self, parsed_html, source):
        links = []
        if source == 'alef':
            elements = parsed_html.find_all('p', {'class': 'mainDiv5Title'})
            for el in elements:
                links.append('http://alef.ir/%s' % el.a['href'])
        elif source == 'irna':
            elements = parsed_html.find_all('div', {'class': 'DataListContainer'})
            for el in elements:
                links.append('http://www.irna.ir/%s' % el.h2.a['href'])
        elif source == 'fars':
            elements = parsed_html.find_all('div', {'class': 'ctgtopnewsinfotitle'})
            elements.extend(parsed_html.find_all('div', {'class': 'ctgimpnewsinfotitle'}))
            for el in elements:
                links.append(el.parent['href'])

        for link in links:
            f = urllib.urlopen(link)
            html = f.read()
            parsed_html = BeautifulSoup(html, 'html.parser')
            res = self.parse_news(parsed_html, source)
            post = self.save_doc(res)
            if post:
                self.stdout.write('doc saved from link: %s' % link)
            else:
                self.stdout.write('NOTICE: doc exists for link: %s' % link)

    def parse_news(self, parsed_html, source):
        author = u'نامعین'

        if source == 'fars':
            body_pane = parsed_html.find('span', id='nwstxtBodyPane')
            text = u''.join(body_pane.find_all('p').strings).strip()
            dt_str = parsed_html.find('div', {'class': 'nwstxtdt'}).text
            (year, month, day, hour, minute) = re.match(r'(\d+)/(\d+)/(\d+) - (\d+):(\d+)', dt_str).groups()
            year = '13' + year
        elif source == 'irna':
            text = parsed_html.find('p',
                                    id='ctl00_ctl00_ContentPlaceHolder_ContentPlaceHolder_NewsContent3_BodyLabel').text
            text = text.replace('<br/>', '\n').strip()

            search_res = re.search(r'\* ?از:(.+) \('.decode('utf-8'), text)
            if search_res:
                author = search_res.groups()[0].strip()
                text = re.sub(r'\* ?از:.+$'.decode('utf-8'), '', text)

            date_str = parsed_html.find('span',
                                        id='ctl00_ctl00_ContentPlaceHolder_ContentPlaceHolder_NewsContent3_NofaDateLabel2').text
            time_str = parsed_html.find('span',
                                        id='ctl00_ctl00_ContentPlaceHolder_ContentPlaceHolder_NewsContent3_NofaDateLabel3').text
            (day, month, year) = re.match(r'(\d+)/(\d+)/(\d+)', date_str).groups()
            (hour, minute) = re.match(r'(\d+):(\d+)', time_str).groups()
        elif source == 'alef':
            text = u''.join(parsed_html.find('div', id='doc_div33').div.strings).strip()
            if not text:
                text = u''.join(parsed_html.find('div', id='doc_div33').strings).strip()
            auth_div = parsed_html.find('div', id='docDiv3TitrSub')
            if auth_div:
                auth_div_text = auth_div.text
                if '-' in auth_div_text:
                    author = auth_div_text.split('-')[1].strip()
                else:
                    author = auth_div_text
                if u'،' in author:
                    author = author.split(u'،')[0].strip()
            datetime_str = parsed_html.find('div', id='docDiv3Date').span.text
            (day, month, year, hour, minute) = [unify_text(group) for group in
                                                re.search(r'(\S+) (\S+) (\S+) ساعت (\S+):(\S+)'.decode('utf-8'),
                                                          datetime_str).groups()]
            month = shamsi_month_num(month)
        else:
            return None

        text = unify_text(text)
        doc_date = jdatetime.date(int(year), int(month), int(day)).togregorian()
        doc_time = datetime.time(int(hour), int(minute))
        doc_dt = datetime.datetime.combine(doc_date, doc_time)
        doc_dt = pytz.timezone(get_default_timezone_name()).localize(doc_dt)
        return {'text': text, 'author': author, 'datetime': doc_dt}

    def save_doc(self, res):
        net_name = u'تحلیل'
        try:
            net = SocialNet.objects.get(name=net_name)
        except ObjectDoesNotExist:
            net = SocialNet.objects.create(name=net_name)
        user = None
        author = res['author']
        if author:
            index = author.find(' ')
            if index > 0:
                first_name = author[:index]
                last_name = author[index + 1:]
            else:
                first_name = author
                last_name = ''
            persons = Person.objects.filter(first_name=first_name, last_name=last_name)
            if persons.exists():
                person = persons[0]
            else:
                person = Person.objects.create(first_name=first_name, last_name=last_name)
            if person.useraccount_set.exists():
                user = persons[0].useraccount_set.all()[0]
            else:
                username = ('%s_%s' % (random.choice(string.letters), random.choice(self.names))).lower()
                user = UserAccount.objects.create(person=person, username=username, social_net=net)

        result = Post.objects.filter(text__icontains=res['text'], datetime=res['datetime'])
        if not result.exists():
            return Post.objects.create(author=user, text=res['text'], datetime=res['datetime'])
        else:
            return None
