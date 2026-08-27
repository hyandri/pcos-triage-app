from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_POST

from accounts.models import AssessmentSession, MedicalReportFile


ALLOWED_REPORT_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_REPORT_SIZE = 10 * 1024 * 1024


@login_required
def clinical_assessment_view(request):
    """Render the client-side Stage 2 clinical assessment form."""
    return render(request, "clinical/assessment.html")


@login_required
@require_POST
def upload_report_view(request):
    """Save a report file against a completed clinical assessment; no OCR is performed."""
    session_id = request.POST.get("assessment_id")
    report = request.FILES.get("report")
    if not session_id or report is None:
        return JsonResponse({"error": "assessment_id and report are required"}, status=400)

    session = get_object_or_404(
        AssessmentSession,
        pk=session_id,
        user=request.user,
        assessment_type=AssessmentSession.AssessmentType.CLINICAL,
    )
    extension = Path(report.name).suffix.lower()
    if extension not in ALLOWED_REPORT_EXTENSIONS:
        return JsonResponse({"error": "Only PDF, JPG, JPEG, and PNG reports are supported"}, status=400)
    if report.size > MAX_REPORT_SIZE:
        return JsonResponse({"error": "Report files must be 10 MB or smaller"}, status=400)

    stored_report = MedicalReportFile.objects.create(
        user=request.user,
        session=session,
        file=report,
        original_filename=report.name,
    )
    return JsonResponse({"report_id": stored_report.pk, "filename": stored_report.original_filename})
