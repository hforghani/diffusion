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

        # Adding model 'UserAccount'
        db.create_table(u'crud_useraccount', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('social_net', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.SocialNet'])),
            ('username', self.gf('django.db.models.fields.CharField')(max_length=100)),
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
            ('datetime', self.gf('django.db.models.fields.DateTimeField')(db_index=True, null=True, blank=True)),
            ('text', self.gf('django.db.models.fields.TextField')(null=True, blank=True)),
            ('url', self.gf('django.db.models.fields.CharField')(db_index=True, max_length=100, null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Post'])

        # Adding model 'Reshare'
        db.create_table(u'crud_reshare', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('post', self.gf('django.db.models.fields.related.ForeignKey')(related_name='parents', to=orm['crud.Post'])),
            ('reshared_post', self.gf('django.db.models.fields.related.ForeignKey')(related_name='children', null=True, to=orm['crud.Post'])),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(blank=True, related_name='parent_reshares', null=True, to=orm['crud.UserAccount'])),
            ('ref_user', self.gf('django.db.models.fields.related.ForeignKey')(blank=True, related_name='children_reshares', null=True, to=orm['crud.UserAccount'])),
            ('datetime', self.gf('django.db.models.fields.DateTimeField')(db_index=True, null=True, blank=True)),
            ('ref_datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Reshare'])

        # Adding model 'Meme'
        db.create_table(u'crud_meme', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('text', self.gf('django.db.models.fields.TextField')(null=True, blank=True)),
            ('count', self.gf('django.db.models.fields.IntegerField')(null=True, blank=True)),
            ('first_time', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
            ('last_time', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Meme'])

        # Adding model 'PostMeme'
        db.create_table(u'crud_postmeme', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('post', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Post'])),
            ('meme', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Meme'])),
        ))
        db.send_create_signal(u'crud', ['PostMeme'])


    def backwards(self, orm):
        # Deleting model 'SocialNet'
        db.delete_table(u'crud_socialnet')

        # Deleting model 'UserAccount'
        db.delete_table(u'crud_useraccount')

        # Deleting model 'Friendship'
        db.delete_table(u'crud_friendship')

        # Deleting model 'Post'
        db.delete_table(u'crud_post')

        # Deleting model 'Reshare'
        db.delete_table(u'crud_reshare')

        # Deleting model 'Meme'
        db.delete_table(u'crud_meme')

        # Deleting model 'PostMeme'
        db.delete_table(u'crud_postmeme')


    models = {
        u'crud.friendship': {
            'Meta': {'object_name': 'Friendship'},
            'end_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'start_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'user1': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'+'", 'to': u"orm['crud.UserAccount']"}),
            'user2': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'+'", 'to': u"orm['crud.UserAccount']"})
        },
        u'crud.meme': {
            'Meta': {'object_name': 'Meme'},
            'count': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            'first_time': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'last_time': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'text': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'})
        },
        u'crud.post': {
            'Meta': {'object_name': 'Post'},
            'author': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.UserAccount']"}),
            'datetime': ('django.db.models.fields.DateTimeField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'text': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'}),
            'url': ('django.db.models.fields.CharField', [], {'db_index': 'True', 'max_length': '100', 'null': 'True', 'blank': 'True'})
        },
        u'crud.postmeme': {
            'Meta': {'object_name': 'PostMeme'},
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'meme': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Meme']"}),
            'post': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.Post']"})
        },
        u'crud.reshare': {
            'Meta': {'object_name': 'Reshare'},
            'datetime': ('django.db.models.fields.DateTimeField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'post': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'parents'", 'to': u"orm['crud.Post']"}),
            'ref_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'ref_user': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'children_reshares'", 'null': 'True', 'to': u"orm['crud.UserAccount']"}),
            'reshared_post': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'children'", 'null': 'True', 'to': u"orm['crud.Post']"}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'parent_reshares'", 'null': 'True', 'to': u"orm['crud.UserAccount']"})
        },
        u'crud.socialnet': {
            'Meta': {'object_name': 'SocialNet'},
            'icon': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '50'})
        },
        u'crud.useraccount': {
            'Meta': {'object_name': 'UserAccount'},
            'avatar': ('django.db.models.fields.CharField', [], {'max_length': '200', 'null': 'True', 'blank': 'True'}),
            'exit_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'friends_count': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'social_net': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['crud.SocialNet']"}),
            'start_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'username': ('django.db.models.fields.CharField', [], {'max_length': '100'})
        }
    }

    complete_apps = ['crud']