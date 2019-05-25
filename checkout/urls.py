from django.urls import path, re_path
from . import views


app_name = 'checkout'

urlpatterns = [
    re_path(r'^carrinho/adicionar/(?P<slug>[\w_-]+)/$', views.create_cartitem,
        name='create_cartitem'
    ),
    re_path(r'^carrinho/$', views.cart_item, name='cart_item')
]