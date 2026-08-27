"""HTTP endpoints for ML inference."""

import json

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from accounts.models import AssessmentSession
from .utils import FeatureValidationError, InferenceError, ModelArtifactError, predict


@login_required
@require_POST
def prediction_view(request):
    """Run a symptom or clinical prediction from a JSON request body."""
    try:
        body = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"error": "Request body must be valid JSON"}, status=400)

    if not isinstance(body, dict):
        return JsonResponse({"error": "Request body must be a JSON object"}, status=400)

    assessment_type = body.get("assessment_type")
    features = body.get("features")
    if assessment_type not in {
        AssessmentSession.AssessmentType.SYMPTOM,
        AssessmentSession.AssessmentType.CLINICAL,
    }:
        return JsonResponse({"error": "assessment_type must be symptom or clinical"}, status=400)
    if not isinstance(features, dict):
        return JsonResponse({"error": "features must be a JSON object"}, status=400)

    try:
        result = predict(assessment_type, features)
    except FeatureValidationError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except ModelArtifactError as exc:
        return JsonResponse({"error": str(exc)}, status=503)
    except InferenceError as exc:
        return JsonResponse({"error": str(exc)}, status=422)

    session = AssessmentSession.objects.create(
        user=request.user,
        assessment_type=assessment_type,
        input_data=features,
        prediction_results=result,
        risk_tier=result["risk_tier"],
    )
    return JsonResponse(
        {
            "assessment_id": session.pk,
            "assessment_type": assessment_type,
            **result,
        },
        status=200,
    )
