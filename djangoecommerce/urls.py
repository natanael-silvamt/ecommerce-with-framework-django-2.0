from django.contrib import admin
from django.urls import re_path, include
from django.contrib.auth import views as auth_views
from django.conf.urls.static import static
from django.conf import settings

from core import views

urlpatterns = [
	re_path(r'^$', views.index, name='index'),
	re_path(r'^contato/$', views.contact, name='contact'),
    re_path(r'^entrar/$', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    re_path(r'^sair/$', auth_views.LogoutView.as_view(next_page='index'), name='logout'),
    re_path(r'^catalogo/', include('catalog.urls', namespace='catalog')),
    re_path(r'^conta/', include('accounts.urls', namespace='accounts')),
    re_path(r'^compras/', include('checkout.urls', namespace='checkout')),
    re_path(r'^render/', include('renderpdf.urls', namespace='renderpdf')),
    re_path(r'^admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)