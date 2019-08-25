from django.urls import path


from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('reco', views.recognition, name='reo'),
    path('err_upload', views.error_upload, name='err_upload'),
]
