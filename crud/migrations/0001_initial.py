# -*- coding: utf-8 -*-
from south.utils import datetime_utils as datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding model 'SocialNet'
        db.create_table(u'crud_socialnet', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('icon', self.gf('django.db.models.fields.CharField')(max_length=200)),
        ))
        db.send_create_signal(u'crud', ['SocialNet'])

        # Adding model 'Education'
        db.create_table(u'crud_education', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('type', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('field', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('degree', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('school', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('start_year', self.gf('django.db.models.fields.IntegerField')()),
            ('end_year', self.gf('django.db.models.fields.IntegerField')()),
        ))
        db.send_create_signal(u'crud', ['Education'])

        # Adding model 'Work'
        db.create_table(u'crud_work', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('place', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('position', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('start_date', self.gf('django.db.models.fields.DateField')(max_length=50)),
            ('end_date', self.gf('django.db.models.fields.DateField')(max_length=50)),
        ))
        db.send_create_signal(u'crud', ['Work'])

        # Adding model 'Person'
        db.create_table(u'crud_person', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('first_name', self.gf('django.db.models.fields.CharField')(max_length=50, null=True, blank=True)),
            ('last_name', self.gf('django.db.models.fields.CharField')(max_length=50, null=True, blank=True)),
            ('gender', self.gf('django.db.models.fields.NullBooleanField')(null=True, blank=True)),
            ('birth_loc', self.gf('django.db.models.fields.CharField')(max_length=50, null=True, blank=True)),
            ('location', self.gf('django.db.models.fields.CharField')(max_length=50, null=True, blank=True)),
            ('language', self.gf('django.db.models.fields.CharField')(max_length=50, null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Person'])

        # Adding M2M table for field educations on 'Person'
        m2m_table_name = db.shorten_name(u'crud_person_educations')
        db.create_table(m2m_table_name, (
            ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True)),
            ('person', models.ForeignKey(orm[u'crud.person'], null=False)),
            ('education', models.ForeignKey(orm[u'crud.education'], null=False))
        ))
        db.create_unique(m2m_table_name, ['person_id', 'education_id'])

        # Adding M2M table for field works on 'Person'
        m2m_table_name = db.shorten_name(u'crud_person_works')
        db.create_table(m2m_table_name, (
            ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True)),
            ('person', models.ForeignKey(orm[u'crud.person'], null=False)),
            ('work', models.ForeignKey(orm[u'crud.work'], null=False))
        ))
        db.create_unique(m2m_table_name, ['person_id', 'work_id'])

        # Adding model 'UserAccount'
        db.create_table(u'crud_useraccount', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('social_net', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.SocialNet'])),
            ('person', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Person'], null=True, blank=True)),
            ('username', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('friends_count', self.gf('django.db.models.fields.IntegerField')(null=True, blank=True)),
            ('start_datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
            ('exit_datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
            ('avatar', self.gf('django.db.models.fields.CharField')(max_length=200, null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['UserAccount'])

        # Adding model 'Friendship'
        db.create_table(u'crud_friendship', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('user1', self.gf('django.db.models.fields.related.ForeignKey')(related_name='+', to=orm['crud.UserAccount'])),
            ('user2', self.gf('django.db.models.fields.related.ForeignKey')(related_name='+', to=orm['crud.UserAccount'])),
            ('start_datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
            ('end_datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Friendship'])

        # Adding model 'Post'
        db.create_table(u'crud_post', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('author', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.UserAccount'])),
            ('datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
            ('text', self.gf('django.db.models.fields.TextField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Post'])

        # Adding model 'Like'
        db.create_table(u'crud_like', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('post', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Post'])),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.UserAccount'])),
            ('datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Like'])

        # Adding model 'Comment'
        db.create_table(u'crud_comment', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('post', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Post'])),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.UserAccount'])),
            ('datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
            ('text', self.gf('django.db.models.fields.TextField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Comment'])

        # Adding model 'Reshare'
        db.create_table(u'crud_reshare', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('post', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Post'])),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.UserAccount'])),
            ('datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
            ('text', self.gf('django.db.models.fields.TextField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Reshare'])

        # Adding model 'Group'
        db.create_table(u'crud_group', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('type', self.gf('django.db.models.fields.CharField')(max_length=50, null=True, blank=True)),
            ('social_net', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.SocialNet'], null=True, blank=True)),
            ('avatar', self.gf('django.db.models.fields.CharField')(max_length=200, null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Group'])

        # Adding model 'GroupMembership'
        db.create_table(u'crud_groupmembership', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('group', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Group'])),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.UserAccount'])),
            ('datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['GroupMembership'])

        # Adding model 'Hashtag'
        db.create_table(u'crud_hashtag', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=50)),
        ))
        db.send_create_signal(u'crud', ['Hashtag'])

        # Adding model 'HashtagPost'
        db.create_table(u'crud_hashtagpost', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('hashtag', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Hashtag'])),
            ('post', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Post'])),
        ))
        db.send_create_signal(u'crud', ['HashtagPost'])

        # Adding model 'Keyword'
        db.create_table(u'crud_keyword', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(unique=True, max_length=50)),
        ))
        db.send_create_signal(u'crud', ['Keyword'])

        # Adding model 'TermFrequency'
        db.create_table(u'crud_termfrequency', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('post', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Post'])),
            ('term', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Keyword'])),
            ('frequency', self.gf('django.db.models.fields.IntegerField')(default=0)),
        ))
        db.send_create_signal(u'crud', ['TermFrequency'])

        # Adding model 'Topic'
        db.create_table(u'crud_topic', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=50)),
        ))
        db.send_create_signal(u'crud', ['Topic'])

        # Adding M2M table for field users on 'Topic'
        m2m_table_name = db.shorten_name(u'crud_topic_users')
        db.create_table(m2m_table_name, (
            ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True)),
            ('topic', models.ForeignKey(orm[u'crud.topic'], null=False)),
            ('useraccount', models.ForeignKey(orm[u'crud.useraccount'], null=False))
        ))
        db.create_unique(m2m_table_name, ['topic_id', 'useraccount_id'])


    def backwards(self, orm):
        # Deleting model 'SocialNet'
        db.delete_table(u'crud_socialnet')

        # Deleting model 'Education'
        db.delete_table(u'crud_education')

        # Deleting model 'Work'
        db.delete_table(u'crud_work')

        # Deleting model 'Person'
        db.delete_table(u'crud_person')

        # Removing M2M table for field educations on 'Person'
        db.delete_table(db.shorten_name(u'crud_person_educations'))

        # Removing M2M table for field works on 'Person'
        db.delete_table(db.shorten_name(u'crud_person_works'))

        # Deleting model 'UserAccount'
        db.delete_table(u'crud_useraccount')

        # Deleting model 'Friendship'
        db.delete_table(u'crud_friendship')

        # Deleting model 'Post'
        db.delete_table(u'crud_post')

        # Deleting model 'Like'
        db.delete_table(u'crud_like')

        # Deleting model 'Comment'
        db.delete_table(u'crud_comment')

        # Deleting model 'Reshare'
        db.delete_table(u'crud_reshare')

        # Deleting model 'Group'
        db.delete_table(u'crud_group')

        # Deleting model 'GroupMembership'
        db.delete_table(u'crud_groupmembership')

        # Deleting model 'Hashtag'
        db.delete_table(u'crud_hashtag')

        # Deleting model 'HashtagPost'
        db.delete_table(u'crud_hashtagpost')

        # Deleting model 'Keyword'
        db.delete_table(u'crud_keyword')

        # Deleting model 'TermFrequency'
        db.delete_table(u'crud_termfrequency')

        # Deleting model 'Topic'
        db.delete_table(u'crud_topic')

        # Removing M2M table for field users on 'Topic'
        db.delete_table(db.shorten_name(u'crud_topic_users'))


    models = {
        u'crud.comment': {
            'Meta': {'object_name': 'Comment'},
            'datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'post': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Post']"}),
            'text': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.UserAccount']"})
        },
        u'crud.education': {
            'Meta': {'object_name': 'Education'},
            'degree': ('django.db.models.fields.CharField', [], {'max_length': '50'}),
            'end_year': ('django.db.models.fields.IntegerField', [], {}),
            'field': ('django.db.models.fields.CharField', [], {'max_length': '50'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'school': ('django.db.models.fields.CharField', [], {'max_length': '50'}),
            'start_year': ('django.db.models.fields.IntegerField', [], {}),
            'type': ('django.db.models.fields.CharField', [], {'max_length': '50'})
        },
        u'crud.friendship': {
            'Meta': {'object_name': 'Friendship'},
            'end_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'start_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'user1': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'+'", 'to': u"orm['crud.UserAccount']"}),
            'user2': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'+'", 'to': u"orm['crud.UserAccount']"})
        },
        u'crud.group': {
            'Meta': {'object_name': 'Group'},
            'avatar': ('django.db.models.fields.CharField', [], {'max_length': '200', 'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '50'}),
            'social_net': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.SocialNet']", 'null': 'True', 'blank': 'True'}),
            'type': ('django.db.models.fields.CharField', [], {'max_length': '50', 'null': 'True', 'blank': 'True'})
        },
        u'crud.groupmembership': {
            'Meta': {'object_name': 'GroupMembership'},
            'datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'group': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Group']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.UserAccount']"})
        },
        u'crud.hashtag': {
            'Meta': {'object_name': 'Hashtag'},
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '50'})
        },
        u'crud.hashtagpost': {
            'Meta': {'object_name': 'HashtagPost'},
            'hashtag': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Hashtag']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'post': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Post']"})
        },
        u'crud.keyword': {
            'Meta': {'object_name': 'Keyword'},
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '50'})
        },
        u'crud.like': {
            'Meta': {'object_name': 'Like'},
            'datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'post': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Post']"}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.UserAccount']"})
        },
        u'crud.person': {
            'Meta': {'object_name': 'Person'},
            'birth_loc': ('django.db.models.fields.CharField', [], {'max_length': '50', 'null': 'True', 'blank': 'True'}),
            'educations': ('django.db.models.fields.related.ManyToManyField', [], {'to': u"orm['crud.Education']", 'symmetrical': 'False'}),
            'first_name': ('django.db.models.fields.CharField', [], {'max_length': '50', 'null': 'True', 'blank': 'True'}),
            'gender': ('django.db.models.fields.NullBooleanField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'language': ('django.db.models.fields.CharField', [], {'max_length': '50', 'null': 'True', 'blank': 'True'}),
            'last_name': ('django.db.models.fields.CharField', [], {'max_length': '50', 'null': 'True', 'blank': 'True'}),
            'location': ('django.db.models.fields.CharField', [], {'max_length': '50', 'null': 'True', 'blank': 'True'}),
            'works': ('django.db.models.fields.related.ManyToManyField', [], {'to': u"orm['crud.Work']", 'symmetrical': 'False'})
        },
        u'crud.post': {
            'Meta': {'object_name': 'Post'},
            'author': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.UserAccount']"}),
            'datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'text': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'})
        },
        u'crud.reshare': {
            'Meta': {'object_name': 'Reshare'},
            'datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'post': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Post']"}),
            'text': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.UserAccount']"})
        },
        u'crud.socialnet': {
            'Meta': {'object_name': 'SocialNet'},
            'icon': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '50'})
        },
        u'crud.termfrequency': {
            'Meta': {'object_name': 'TermFrequency'},
            'frequency': ('django.db.models.fields.IntegerField', [], {'default': '0'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'post': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Post']"}),
            'term': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Keyword']"})
        },
        u'crud.topic': {
            'Meta': {'object_name': 'Topic'},
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '50'}),
            'users': ('django.db.models.fields.related.ManyToManyField', [], {'to': u"orm['crud.UserAccount']", 'symmetrical': 'False'})
        },
        u'crud.useraccount': {
            'Meta': {'object_name': 'UserAccount'},
            'avatar': ('django.db.models.fields.CharField', [], {'max_length': '200', 'null': 'True', 'blank': 'True'}),
            'exit_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'friends_count': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'person': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Person']", 'null': 'True', 'blank': 'True'}),
            'social_net': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.SocialNet']"}),
            'start_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'username': ('django.db.models.fields.CharField', [], {'max_length': '50'})
        },
        u'crud.work': {
            'Meta': {'object_name': 'Work'},
            'end_date': ('django.db.models.fields.DateField', [], {'max_length': '50'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'place': ('django.db.models.fields.CharField', [], {'max_length': '50'}),
            'position': ('django.db.models.fields.CharField', [], {'max_length': '50'}),
            'start_date': ('django.db.models.fields.DateField', [], {'max_length': '50'})
        }
    }

    complete_apps = ['crud']