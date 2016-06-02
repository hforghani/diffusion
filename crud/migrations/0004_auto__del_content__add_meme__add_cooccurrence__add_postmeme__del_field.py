# -*- coding: utf-8 -*-
from south.utils import datetime_utils as datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Deleting model 'Content'
        db.delete_table(u'crud_content')

        # Adding model 'Meme'
        db.create_table(u'crud_meme', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('text', self.gf('django.db.models.fields.TextField')(null=True, blank=True)),
            ('count', self.gf('django.db.models.fields.IntegerField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Meme'])

        # Adding model 'CoOccurrence'
        db.create_table(u'crud_cooccurrence', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('term1', self.gf('django.db.models.fields.related.ForeignKey')(related_name='+', to=orm['crud.Keyword'])),
            ('term2', self.gf('django.db.models.fields.related.ForeignKey')(related_name='+', to=orm['crud.Keyword'])),
            ('count', self.gf('django.db.models.fields.IntegerField')(default=0)),
        ))
        db.send_create_signal(u'crud', ['CoOccurrence'])

        # Adding model 'PostMeme'
        db.create_table(u'crud_postmeme', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('post', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Post'])),
            ('meme', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Meme'])),
        ))
        db.send_create_signal(u'crud', ['PostMeme'])

        # Deleting field 'Reshare.text'
        db.delete_column(u'crud_reshare', 'text')

        # Adding field 'Reshare.reshared_post'
        db.add_column(u'crud_reshare', 'reshared_post',
                      self.gf('django.db.models.fields.related.ForeignKey')(related_name='reshares', null=True, to=orm['crud.Post']),
                      keep_default=False)

        # Deleting field 'Post.content'
        db.delete_column(u'crud_post', 'content_id')


    def backwards(self, orm):
        # Adding model 'Content'
        db.create_table(u'crud_content', (
            ('text', self.gf('django.db.models.fields.TextField')(null=True, blank=True)),
            ('shares', self.gf('django.db.models.fields.IntegerField')(null=True, blank=True)),
            ('comments', self.gf('django.db.models.fields.IntegerField')(null=True, blank=True)),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('likes', self.gf('django.db.models.fields.IntegerField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Content'])

        # Deleting model 'Meme'
        db.delete_table(u'crud_meme')

        # Deleting model 'CoOccurrence'
        db.delete_table(u'crud_cooccurrence')

        # Deleting model 'PostMeme'
        db.delete_table(u'crud_postmeme')

        # Adding field 'Reshare.text'
        db.add_column(u'crud_reshare', 'text',
                      self.gf('django.db.models.fields.TextField')(null=True, blank=True),
                      keep_default=False)

        # Deleting field 'Reshare.reshared_post'
        db.delete_column(u'crud_reshare', 'reshared_post_id')

        # Adding field 'Post.content'
        db.add_column(u'crud_post', 'content',
                      self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Content'], null=True, blank=True),
                      keep_default=False)


    models = {
        u'crud.comment': {
            'Meta': {'object_name': 'Comment'},
            'datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'post': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Post']"}),
            'text': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.UserAccount']"})
        },
        u'crud.cooccurrence': {
            'Meta': {'object_name': 'CoOccurrence'},
            'count': ('django.db.models.fields.IntegerField', [], {'default': '0'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'term1': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'+'", 'to': u"orm['crud.Keyword']"}),
            'term2': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'+'", 'to': u"orm['crud.Keyword']"})
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
            'idf': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '50'})
        },
        u'crud.like': {
            'Meta': {'object_name': 'Like'},
            'datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'post': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Post']"}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.UserAccount']"})
        },
        u'crud.meme': {
            'Meta': {'object_name': 'Meme'},
            'count': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'text': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'})
        },
        u'crud.person': {
            'Meta': {'object_name': 'Person'},
            'birth_loc': ('django.db.models.fields.CharField', [], {'max_length': '50', 'null': 'True', 'blank': 'True'}),
            'educations': ('django.db.models.fields.related.ManyToManyField', [], {'symmetrical': 'False', 'to': u"orm['crud.Education']", 'null': 'True', 'blank': 'True'}),
            'first_name': ('django.db.models.fields.CharField', [], {'max_length': '50', 'null': 'True', 'blank': 'True'}),
            'gender': ('django.db.models.fields.NullBooleanField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'language': ('django.db.models.fields.CharField', [], {'max_length': '50', 'null': 'True', 'blank': 'True'}),
            'last_name': ('django.db.models.fields.CharField', [], {'max_length': '50', 'null': 'True', 'blank': 'True'}),
            'location': ('django.db.models.fields.CharField', [], {'max_length': '50', 'null': 'True', 'blank': 'True'}),
            'works': ('django.db.models.fields.related.ManyToManyField', [], {'symmetrical': 'False', 'to': u"orm['crud.Work']", 'null': 'True', 'blank': 'True'})
        },
        u'crud.post': {
            'Meta': {'object_name': 'Post'},
            'author': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.UserAccount']"}),
            'datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'text': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'})
        },
        u'crud.postmeme': {
            'Meta': {'object_name': 'PostMeme'},
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'meme': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Meme']"}),
            'post': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Post']"})
        },
        u'crud.reshare': {
            'Meta': {'object_name': 'Reshare'},
            'datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'post': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'reshared_from'", 'to': u"orm['crud.Post']"}),
            'reshared_post': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'reshares'", 'null': 'True', 'to': u"orm['crud.Post']"}),
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