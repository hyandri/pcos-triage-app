from django.urls import path

from . import views


app_name = "clinical"

urlpatterns = [
    path("", views.clinical_assessment_view, name="assessment"),
    path("upload-report/", views.upload_report_view, name="upload_report"),
]
