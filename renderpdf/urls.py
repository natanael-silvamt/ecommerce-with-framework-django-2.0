from django.urls import path, re_path
from . import views


app_name = 'renderpdf'

urlpatterns = [
    re_path(r'^pdf/$', views.pdf, name='render_pdf'),
    re_path(r'^pdf-date/$', views.pdf_date, name='render_pdf_date'),
    re_path(r'^chart-pedidos/$', views.render_chart, name='render_chart'),
    re_path(r'^predict/$', views.predict, name='predict'),
]