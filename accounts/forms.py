# -*- coding: utf-8 -*-
from django import forms
from django.contrib.auth import authenticate

__docformat__ = 'reStructuredText'


class LoginForm(forms.Form):
    username = forms.CharField(max_length=80, label=u'نام کاربری')
    password = forms.CharField(max_length=80, label=u'رمز عبور', widget=forms.PasswordInput())

    user = None

    def clean(self):
        cleaned_data = super(LoginForm, self).clean()
        username = cleaned_data.get('username', None)
        password = cleaned_data.get('password', None)
        user = authenticate(username=username, password=password)
        if user is not None:
            if not user.is_active:
                raise forms.ValidationError(u"شما یک کاربر فعال نیستید")
            self.user = user
        else:
            raise forms.ValidationError(u"نام کاربری یا رمز عبور اشتباه است")
        return cleaned_data
