from django.contrib.auth.decorators import login_required
from django.shortcuts import render


@login_required
def symptom_flashcards_view(request):
    """Render the client-side Stage 1 symptom assessment."""
    return render(request, "triage/flashcards.html")
