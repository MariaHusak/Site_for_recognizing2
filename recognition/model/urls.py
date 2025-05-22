from django.urls import path
from . import views

urlpatterns = [
    path('select/', views.select_file, name='select_file'),
    path('segment/<int:file_id>/', views.segment_file, name='segment_file'),
    path('download/<path:file_path>/', views.download_video, name='download_video'),
]