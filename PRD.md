# PRODUCT REQUIREMENTS DOCUMENT (PRD)

## 1. Product Vision
A dual-stage web application to assess PCOS risk. It provides a frictionless, interactive flashcard UI for symptom tracking, and a structured clinical data entry flow for users with medical reports. All data is persisted to user accounts to build a longitudinal health profile.

## 2. User Personas
- **Persona A (Curious/At-Risk):** Wants a quick, anonymous-ish (but tracked) check on their symptoms.
- **Persona B (Clinical):** Has recent bloodwork/ultrasounds and wants a highly accurate, data-backed risk assessment.

## 3. Core Features (MVP Scope)
### Phase 1: Authentication & Dashboard
- User Registration/Login (Django Auth).
- Dashboard showing past assessment history (list of `AssessmentSession` objects).
- Ability to click into past assessments to see the inputs and results.

### Phase 2: Stage 1 Triage (Symptom Flashcards)
- Landing page with two clear choices: "Quick Symptom Check" vs "Full Clinical Assessment".
- Interactive HTMX/Alpine.js flashcards for the 10 symptom features.
- Progress bar (e.g., "Step 3 of 10").
- Immediate result generation and saving to DB.

### Phase 3: Stage 2 Clinical Assessment
- Form-based UI grouped logically: 
  1. Ultrasound Markers
  2. Blood & Hormone Panels
- "Upload Report" feature: Allows user to upload a PDF/Image. Saves to `MedicalReportFile` model. (No OCR parsing in MVP).
- Combines Stage 1 symptoms + Stage 2 medical data to run Model B.

## 4. Out of Scope for MVP
- Automated OCR parsing of medical reports.
- LLM-generated plain English explanations (SHAP value logic is saved to DB, but UI for it is not built yet).
- Diet/Lifestyle recommendation engine.