from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('main', views.ocr_main, name='ocr_main'),
]