from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('main', views.parse_single_man_zt, name='parse_single_man'),
    path('err_upload', views.error_upload, name='err_upload'),
]