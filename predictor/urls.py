# predictor/urls.py
from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    # Example: http://127.0.0.1:8000/
    path('', views.home, name='home'),
    
    # Example: http://127.0.0.1:8000/single/
    path('single/', views.single_prediction_view, name='single_prediction'),
    
    # Example: http://127.0.0.1:8000/batch/
    path('batch/', views.batch_prediction_view, name='batch_prediction'),
]