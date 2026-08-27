from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update(
                {
                    'class': (
                        'mt-1 block w-full rounded-lg border border-slate-300 '
                        'px-3 py-2 shadow-sm focus:border-rose-500 '
                        'focus:outline-none focus:ring-1 focus:ring-rose-500'
                    ),
                }
            )

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user
