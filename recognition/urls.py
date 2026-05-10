from django.urls import path
from recognition.views import predict

urlpatterns = [
    path('predict/', predict),
]