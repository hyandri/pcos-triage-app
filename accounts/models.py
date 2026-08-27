from django.conf import settings
from django.db import models


class UserProfile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='profile',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'Profile for {self.user.username}'


class AssessmentSession(models.Model):
    class AssessmentType(models.TextChoices):
        SYMPTOM = 'symptom', 'Symptom Triage'
        CLINICAL = 'clinical', 'Full Clinical Assessment'

    class RiskTier(models.TextChoices):
        LOW = 'low', 'Low Risk'
        MODERATE = 'moderate', 'Moderate Risk'
        HIGH = 'high', 'High Risk'
        PENDING = 'pending', 'Pending'

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='assessment_sessions',
    )
    assessment_type = models.CharField(
        max_length=20,
        choices=AssessmentType.choices,
    )
    input_data = models.JSONField(
        default=dict,
        blank=True,
        help_text='Raw feature inputs collected during the assessment.',
    )
    prediction_results = models.JSONField(
        default=dict,
        blank=True,
        help_text='Model output including probability scores and metadata.',
    )
    shap_values = models.JSONField(
        default=dict,
        blank=True,
        help_text='Placeholder for future SHAP explainability data.',
    )
    risk_tier = models.CharField(
        max_length=20,
        choices=RiskTier.choices,
        default=RiskTier.PENDING,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.get_assessment_type_display()} — {self.user.username} ({self.created_at:%Y-%m-%d})'


class MedicalReportFile(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='medical_report_files',
    )
    session = models.ForeignKey(
        AssessmentSession,
        on_delete=models.CASCADE,
        related_name='report_files',
    )
    file = models.FileField(upload_to='medical_reports/%Y/%m/%d/')
    original_filename = models.CharField(max_length=255, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return self.original_filename or self.file.name

    def save(self, *args, **kwargs):
        if self.file and not self.original_filename:
            self.original_filename = self.file.name
        super().save(*args, **kwargs)
