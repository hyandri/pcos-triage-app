from django.urls import path

from .views import prediction_view


app_name = "ml_engine"

urlpatterns = [
    path("predict/", prediction_view, name="predict"),
]
