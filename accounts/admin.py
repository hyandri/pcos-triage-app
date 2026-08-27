from django.contrib import admin

from .models import AssessmentSession, MedicalReportFile, UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'created_at', 'updated_at')
    search_fields = ('user__username', 'user__email')


@admin.register(AssessmentSession)
class AssessmentSessionAdmin(admin.ModelAdmin):
    list_display = (
        'user',
        'assessment_type',
        'risk_tier',
        'created_at',
    )
    list_filter = ('assessment_type', 'risk_tier')
    search_fields = ('user__username',)
    readonly_fields = ('created_at', 'updated_at')


@admin.register(MedicalReportFile)
class MedicalReportFileAdmin(admin.ModelAdmin):
    list_display = ('original_filename', 'user', 'session', 'uploaded_at')
    search_fields = ('original_filename', 'user__username')
