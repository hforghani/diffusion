# -*- coding: utf-8 -*-
from south.utils import datetime_utils as datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Deleting model 'Education'
        db.delete_table(u'crud_education')

        # Deleting model 'Group'
        db.delete_table(u'crud_group')

        # Deleting model 'Hashtag'
        db.delete_table(u'crud_hashtag')

        # Deleting model 'Keyword'
        db.delete_table(u'crud_keyword')

        # Deleting model 'DataEntry'
        db.delete_table(u'crud_dataentry')

        # Deleting model 'Work'
        db.delete_table(u'crud_work')

        # Deleting model 'Comment'
        db.delete_table(u'crud_comment')

        # Deleting model 'ProjectMembership'
        db.delete_table(u'crud_projectmembership')

        # Deleting model 'TermFrequency'
        db.delete_table(u'crud_termfrequency')

        # Deleting model 'TimeStep'
        db.delete_table(u'crud_timestep')

        # Deleting model 'Like'
        db.delete_table(u'crud_like')

        # Deleting model 'Topic'
        db.delete_table(u'crud_topic')

        # Removing M2M table for field users on 'Topic'
        db.delete_table(db.shorten_name(u'crud_topic_users'))

        # Deleting model 'Project'
        db.delete_table(u'crud_project')

        # Deleting model 'CoOccurrence'
        db.delete_table(u'crud_cooccurrence')

        # Deleting model 'HashtagPost'
        db.delete_table(u'crud_hashtagpost')

        # Deleting model 'DiffusionParam'
        db.delete_table(u'crud_diffusionparam')

        # Deleting model 'UserTermFreq'
        db.delete_table(u'crud_usertermfreq')

        # Deleting model 'GroupMembership'
        db.delete_table(u'crud_groupmembership')

        # Deleting model 'Person'
        db.delete_table(u'crud_person')

        # Removing M2M table for field educations on 'Person'
        db.delete_table(db.shorten_name(u'crud_person_educations'))

        # Removing M2M table for field works on 'Person'
        db.delete_table(db.shorten_name(u'crud_person_works'))

        # Deleting field 'UserAccount.person'
        db.delete_column(u'crud_useraccount', 'person_id')


    def backwards(self, orm):
        # Adding model 'Education'
        db.create_table(u'crud_education', (
            ('field', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('school', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('degree', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('start_year', self.gf('django.db.models.fields.IntegerField')()),
            ('end_year', self.gf('django.db.models.fields.IntegerField')()),
            ('type', self.gf('django.db.models.fields.CharField')(max_length=50)),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
        ))
        db.send_create_signal(u'crud', ['Education'])

        # Adding model 'Group'
        db.create_table(u'crud_group', (
            ('name', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('social_net', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.SocialNet'], null=True, blank=True)),
            ('avatar', self.gf('django.db.models.fields.CharField')(max_length=200, null=True, blank=True)),
            ('type', self.gf('django.db.models.fields.CharField')(max_length=50, null=True, blank=True)),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
        ))
        db.send_create_signal(u'crud', ['Group'])

        # Adding model 'Hashtag'
        db.create_table(u'crud_hashtag', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=255)),
        ))
        db.send_create_signal(u'crud', ['Hashtag'])

        # Adding model 'Keyword'
        db.create_table(u'crud_keyword', (
            ('idf', self.gf('django.db.models.fields.FloatField')(null=True, blank=True)),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=50, unique=True)),
        ))
        db.send_create_signal(u'crud', ['Keyword'])

        # Adding model 'DataEntry'
        db.create_table(u'crud_dataentry', (
            ('comments_message', self.gf('django.db.models.fields.CharField')(max_length=500)),
            ('posts_message', self.gf('django.db.models.fields.CharField')(max_length=500)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('friends_progress', self.gf('django.db.models.fields.IntegerField')(default=0)),
            ('started', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('comments_progress', self.gf('django.db.models.fields.IntegerField')(default=0)),
            ('reshares_progress', self.gf('django.db.models.fields.IntegerField')(default=0)),
            ('nets_message', self.gf('django.db.models.fields.CharField')(max_length=500)),
            ('finished', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('friends_message', self.gf('django.db.models.fields.CharField')(max_length=500)),
            ('likes_progress', self.gf('django.db.models.fields.IntegerField')(default=0)),
            ('reshares_message', self.gf('django.db.models.fields.CharField')(max_length=500)),
            ('users_message', self.gf('django.db.models.fields.CharField')(max_length=500)),
            ('likes_message', self.gf('django.db.models.fields.CharField')(max_length=500)),
            ('posts_progress', self.gf('django.db.models.fields.IntegerField')(default=0)),
            ('users_progress', self.gf('django.db.models.fields.IntegerField')(default=0)),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('nets_progress', self.gf('django.db.models.fields.IntegerField')(default=0)),
        ))
        db.send_create_signal(u'crud', ['DataEntry'])

        # Adding model 'Work'
        db.create_table(u'crud_work', (
            ('end_date', self.gf('django.db.models.fields.DateField')(max_length=50)),
            ('start_date', self.gf('django.db.models.fields.DateField')(max_length=50)),
            ('place', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('position', self.gf('django.db.models.fields.CharField')(max_length=50)),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
        ))
        db.send_create_signal(u'crud', ['Work'])

        # Adding model 'Comment'
        db.create_table(u'crud_comment', (
            ('text', self.gf('django.db.models.fields.TextField')(null=True, blank=True)),
            ('datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.UserAccount'])),
            ('post', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Post'])),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
        ))
        db.send_create_signal(u'crud', ['Comment'])

        # Adding model 'ProjectMembership'
        db.create_table(u'crud_projectmembership', (
            ('project', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Project'])),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.UserAccount'])),
        ))
        db.send_create_signal(u'crud', ['ProjectMembership'])

        # Adding model 'TermFrequency'
        db.create_table(u'crud_termfrequency', (
            ('term', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Keyword'])),
            ('post', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Post'])),
            ('frequency', self.gf('django.db.models.fields.IntegerField')(default=0)),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
        ))
        db.send_create_signal(u'crud', ['TermFrequency'])

        # Adding model 'TimeStep'
        db.create_table(u'crud_timestep', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('datetime', self.gf('django.db.models.fields.DateTimeField')()),
        ))
        db.send_create_signal(u'crud', ['TimeStep'])

        # Adding model 'Like'
        db.create_table(u'crud_like', (
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.UserAccount'])),
            ('post', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Post'])),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['Like'])

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

        # Adding model 'Project'
        db.create_table(u'crud_project', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=50)),
        ))
        db.send_create_signal(u'crud', ['Project'])

        # Adding model 'CoOccurrence'
        db.create_table(u'crud_cooccurrence', (
            ('count', self.gf('django.db.models.fields.IntegerField')(default=0, db_index=True)),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('term1', self.gf('django.db.models.fields.related.ForeignKey')(related_name='+', to=orm['crud.Keyword'])),
            ('term2', self.gf('django.db.models.fields.related.ForeignKey')(related_name='+', to=orm['crud.Keyword'])),
        ))
        db.send_create_signal(u'crud', ['CoOccurrence'])

        # Adding model 'HashtagPost'
        db.create_table(u'crud_hashtagpost', (
            ('post', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Post'])),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('hashtag', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Hashtag'])),
        ))
        db.send_create_signal(u'crud', ['HashtagPost'])

        # Adding model 'DiffusionParam'
        db.create_table(u'crud_diffusionparam', (
            ('sender', self.gf('django.db.models.fields.related.ForeignKey')(related_name='params_from', to=orm['crud.UserAccount'])),
            ('weight', self.gf('django.db.models.fields.FloatField')(default=0)),
            ('delay', self.gf('django.db.models.fields.FloatField')(default=0)),
            ('receiver', self.gf('django.db.models.fields.related.ForeignKey')(related_name='params_to', to=orm['crud.UserAccount'])),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
        ))
        db.send_create_signal(u'crud', ['DiffusionParam'])

        # Adding model 'UserTermFreq'
        db.create_table(u'crud_usertermfreq', (
            ('term', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Keyword'])),
            ('to_datetime', self.gf('django.db.models.fields.DateTimeField')(null=True)),
            ('frequency', self.gf('django.db.models.fields.IntegerField')(default=0)),
            ('from_datetime', self.gf('django.db.models.fields.DateTimeField')(null=True)),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.UserAccount'])),
        ))
        db.send_create_signal(u'crud', ['UserTermFreq'])

        # Adding model 'GroupMembership'
        db.create_table(u'crud_groupmembership', (
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.UserAccount'])),
            ('group', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Group'])),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'crud', ['GroupMembership'])

        # Adding model 'Person'
        db.create_table(u'crud_person', (
            ('bio', self.gf('django.db.models.fields.TextField')(null=True, blank=True)),
            ('last_name', self.gf('django.db.models.fields.CharField')(max_length=50, null=True, blank=True)),
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('first_name', self.gf('django.db.models.fields.CharField')(max_length=50, null=True, blank=True)),
            ('language', self.gf('django.db.models.fields.CharField')(max_length=50, null=True, blank=True)),
            ('gender', self.gf('django.db.models.fields.NullBooleanField')(null=True, blank=True)),
            ('birth_loc', self.gf('django.db.models.fields.CharField')(max_length=50, null=True, blank=True)),
            ('location', self.gf('django.db.models.fields.CharField')(max_length=50, null=True, blank=True)),
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

        # Adding field 'UserAccount.person'
        db.add_column(u'crud_useraccount', 'person',
                      self.gf('django.db.models.fields.related.ForeignKey')(to=orm['crud.Person'], null=True, blank=True),
                      keep_default=False)


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