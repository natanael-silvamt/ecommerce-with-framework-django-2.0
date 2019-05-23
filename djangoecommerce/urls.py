"""djangoecommerce URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import re_path, include
from django.contrib.auth import views as auth_views

from core import views

urlpatterns = [
	re_path(r'^$', views.index, name='index'),
	re_path(r'^contato/$', views.contact, name='contact'),
    re_path(r'^entrar/$', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    re_path(r'^sair/$', auth_views.LogoutView.as_view(next_page='index'), name='logout'),
    re_path(r'^catalogo/', include('catalog.urls', namespace='catalog')),
    re_path(r'^conta/', include('accounts.urls', namespace='accounts')),
    re_path(r'^compras/', include('checkout.urls', namespace='checkout')),
    re_path(r'^admin/', admin.site.urls),
]
