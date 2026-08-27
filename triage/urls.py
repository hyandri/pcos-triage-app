from django.urls import path

from . import views


app_name = "triage"

urlpatterns = [
    path("symptoms/", views.symptom_flashcards_view, name="symptom_flashcards"),
]
