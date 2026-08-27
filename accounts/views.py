from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.shortcuts import redirect, render
from django.urls import reverse_lazy
from django.views.generic import CreateView

from .forms import RegisterForm
from .models import AssessmentSession, UserProfile


class RegisterView(CreateView):
    form_class = RegisterForm
    template_name = 'accounts/register.html'
    success_url = reverse_lazy('accounts:dashboard')

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('accounts:dashboard')
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        response = super().form_valid(form)
        UserProfile.objects.get_or_create(user=self.object)
        login(self.request, self.object)
        messages.success(self.request, 'Welcome! Your account has been created.')
        return response


class CustomLoginView(LoginView):
    template_name = 'accounts/login.html'
    redirect_authenticated_user = True

    def get_success_url(self):
        return reverse_lazy('accounts:dashboard')


class CustomLogoutView(LogoutView):
    next_page = reverse_lazy('accounts:login')


@login_required
def dashboard_view(request):
    assessments = AssessmentSession.objects.filter(user=request.user)
    return render(
        request,
        'accounts/dashboard.html',
        {'assessments': assessments},
    )
